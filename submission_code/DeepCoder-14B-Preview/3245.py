import bisect

class Solution:
    def beautifulIndices(self, s: str, a: str, b: str, k: int) -> List[int]:
        # Helper function to find all starting indices of a substring
        def find_occurrences(sub):
            indices = []
            len_sub = len(sub)
            for i in range(len(s) - len_sub + 1):
                if s[i:i+len_sub] == sub:
                    indices.append(i)
            return indices
        
        # Find all starting indices for substrings a and b
        A = find_occurrences(a)
        B = find_occurrences(b)
        
        # Sort B to enable binary search
        B.sort()
        
        result = []
        for i in A:
            low = i - k
            high = i + k
            # Find the first index in B that is >= low
            left_pos = bisect.bisect_left(B, low)
            # Check if this index is within the high boundary
            if left_pos < len(B) and B[left_pos] <= high:
                result.append(i)
        
        return result