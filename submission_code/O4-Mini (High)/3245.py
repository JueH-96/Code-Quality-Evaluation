from typing import List

class Solution:
    def beautifulIndices(self, s: str, a: str, b: str, k: int) -> List[int]:
        n = len(s)
        la, lb = len(a), len(b)
        # Maximum starting indices for a and b
        max_i = n - la
        max_j = n - lb
        
        # Mark all starting positions of b in s
        is_b = [0] * (max_j + 1)
        for j in range(max_j + 1):
            if s[j:j+lb] == b:
                is_b[j] = 1
        
        # Build prefix sums over is_b for O(1) range queries
        prefix_b = [0] * (max_j + 2)
        for j in range(max_j + 1):
            prefix_b[j+1] = prefix_b[j] + is_b[j]
        
        res = []
        # For each position i where a occurs, check if any b occurs within k
        for i in range(max_i + 1):
            if s[i:i+la] != a:
                continue
            low = max(0, i - k)
            high = min(max_j, i + k)
            # If there's any j in [low, high] with is_b[j] == 1
            if prefix_b[high+1] - prefix_b[low] > 0:
                res.append(i)
        
        return res