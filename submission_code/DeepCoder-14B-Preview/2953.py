from collections import defaultdict

class Solution:
    def countPairs(self, coordinates: List[List[int]], k: int) -> int:
        freq = defaultdict(int)
        for x, y in coordinates:
            freq[(x, y)] += 1
        
        total = 0
        for (x, y), cnt in freq.items():
            for a in range(k + 1):
                b = k - a
                x_prime = x ^ a
                y_prime = y ^ b
                total += freq.get((x_prime, y_prime), 0)
        
        same_pairs_code = 0
        same_pairs_correct = 0
        if k == 0:
            same_pairs_code = sum(cnt for (x, y), cnt in freq.items())
            same_pairs_correct = sum(cnt * (cnt - 1) // 2 for (x, y), cnt in freq.items())
        
        cross_pairs = (total - same_pairs_code) // 2
        
        return same_pairs_correct + cross_pairs