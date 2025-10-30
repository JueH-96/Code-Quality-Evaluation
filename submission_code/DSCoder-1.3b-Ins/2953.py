class Solution:
    def countPairs(self, coordinates: List[List[int]], k: int) -> int:
        from collections import defaultdict
        pair_counts = defaultdict(int)
        pairs_found = 0
        for i in range(len(coordinates)):
            for j in range(i + 1, len(coordinates)):
                x1, y1 = coordinates[i]
                x2, y2 = coordinates[j]
                distance = abs(x1 - x2) + abs(y1 - y2)
                pair_counts[distance] += 1
                if k == 0:
                    pairs_found += pair_counts[distance]
                elif k in pair_counts:
                    pairs_found += pair_counts[k]
                    if k != distance:
                        pairs_found += pair_counts[distance]
        return pairs_found