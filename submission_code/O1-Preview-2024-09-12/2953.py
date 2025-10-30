class Solution:
    def countPairs(self, coordinates: List[List[int]], k: int) -> int:
        from collections import Counter
        count = Counter()
        for x, y in coordinates:
            count[(x, y)] += 1
        
        total_pairs = 0
        for x, y in coordinates:
            for a in range(k + 1):
                b = k - a
                xj = x ^ a
                yj = y ^ b
                cnt = count.get((xj, yj), 0)
                if xj == x and yj == y:
                    cnt -= 1
                total_pairs += cnt
        return total_pairs // 2