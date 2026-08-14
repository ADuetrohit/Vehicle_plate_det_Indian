from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import imagehash
from PIL import Image

from .records import ImageRecord


class _UnionFind:
    def __init__(self, values: list[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, first: str, second: str) -> None:
        left, right = self.find(first), self.find(second)
        if left != right:
            smaller, larger = sorted((left, right))
            self.parent[larger] = smaller


@dataclass
class _BKNode:
    value: int
    record_ids: list[str] = field(default_factory=list)
    children: dict[int, "_BKNode"] = field(default_factory=dict)

    def add(self, value: int, record_id: str) -> None:
        distance = (self.value ^ value).bit_count()
        if distance == 0:
            self.record_ids.append(record_id)
            return
        child = self.children.get(distance)
        if child is None:
            self.children[distance] = _BKNode(value, [record_id])
        else:
            child.add(value, record_id)

    def search(self, value: int, radius: int) -> list[str]:
        distance = (self.value ^ value).bit_count()
        matches = list(self.record_ids) if distance <= radius else []
        for edge, child in self.children.items():
            if distance - radius <= edge <= distance + radius:
                matches.extend(child.search(value, radius))
        return matches


def perceptual_family(
    records: list[ImageRecord], max_distance: int = 4
) -> dict[str, str]:
    if max_distance < 0:
        raise ValueError("max_distance must be non-negative")
    union_find = _UnionFind([record.record_id for record in records])
    tree: _BKNode | None = None
    sequences: dict[str, list[str]] = defaultdict(list)
    for record in records:
        with Image.open(record.image_path) as image:
            hash_value = int(str(imagehash.phash(image)), 16)
        if tree is None:
            tree = _BKNode(hash_value, [record.record_id])
        else:
            for match in tree.search(hash_value, max_distance):
                union_find.union(record.record_id, match)
            tree.add(hash_value, record.record_id)
        sequence_id = record.tags.get("sequence_id")
        if sequence_id:
            sequences[sequence_id].append(record.record_id)
    for members in sequences.values():
        for record_id_value in members[1:]:
            union_find.union(members[0], record_id_value)
    groups: dict[str, list[str]] = defaultdict(list)
    for record in records:
        groups[union_find.find(record.record_id)].append(record.record_id)
    canonical = {
        member: min(members) for members in groups.values() for member in members
    }
    return canonical

