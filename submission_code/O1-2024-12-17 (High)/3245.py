class Solution:
    def beautifulIndices(self, s: str, a: str, b: str, k: int) -> List[int]:
        from typing import List
        
        def get_positions(text: str, pattern: str) -> List[int]:
            positions = []
            plen = len(pattern)
            limit = len(text) - plen + 1
            for i in range(limit):
                if text[i:i+plen] == pattern:
                    positions.append(i)
            return positions
        
        # Find all occurrences of a and b
        A = get_positions(s, a)  # indices where s[i:i+len(a)] == a
        B = get_positions(s, b)  # indices where s[j:j+len(b)] == b
        
        result = []
        p = 0
        lenB = len(B)
        
        # Two-pointer approach: for each i in A, move pointer p in B
        for i in A:
            # Advance p to skip indices that are too far (B[p] < i-k)
            while p < lenB and B[p] < i - k:
                p += 1
            # Now check if B[p] is within i + k
            if p < lenB and B[p] <= i + k:
                result.append(i)
        
        return result