class Solution:
    def countPairs(self, coordinates: List[List[int]], k: int) -> int:
        from collections import Counter
        
        # Frequency map of all points
        freq = Counter(tuple(pt) for pt in coordinates)
        
        # Case when k = 0: distance is 0 only if points are identical
        if k == 0:
            # For each coordinate c with frequency f, number of pairs is f*(f-1)//2
            return sum(f * (f - 1) // 2 for f in freq.values())
        
        # Case when k > 0
        # We will sum up freq[x, y] * freq[x^A, y^(k-A)] for A in [0..k]
        # Then we'll divide by 2 to avoid double-counting (i,j) and (j,i).
        
        total_pairs = 0
        # Precompute all (A, B) such that A + B = k
        AB_pairs = [(A, k - A) for A in range(k + 1)]
        
        for (x, y), count_xy in freq.items():
            for A, B in AB_pairs:
                x2 = x ^ A
                y2 = y ^ B
                # If (x2, y2) exists, contribute count_xy * freq[x2, y2]
                count_x2y2 = freq.get((x2, y2), 0)
                total_pairs += count_xy * count_x2y2
        
        # Since we've counted each pair twice, divide by 2
        return total_pairs // 2