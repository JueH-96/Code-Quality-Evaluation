class Solution:
    def minimumString(self, a: str, b: str, c: str) -> str:
        def merge(s1: str, s2: str) -> str:
            if s1 in s2:
                return s2
            if s2 in s1:
                return s1
            
            n1, n2 = len(s1), len(s2)
            # Try all possible overlaps
            min_len = float('inf')
            result = s1 + s2  # Default: no overlap
            
            for i in range(min(n1, n2)):
                # Check if suffix of s1 matches prefix of s2
                if s1[n1-i-1:] == s2[:i+1]:
                    merged = s1 + s2[i+1:]
                    if len(merged) < min_len:
                        min_len = len(merged)
                        result = merged
                    elif len(merged) == min_len and merged < result:
                        result = merged
            
            return result
        
        # Try all possible orderings of strings
        strings = [a, b, c]
        min_len = float('inf')
        ans = "z" * 300  # Initialize with a large string
        
        from itertools import permutations
        for p in permutations(strings):
            temp = merge(merge(p[0], p[1]), p[2])
            if len(temp) < min_len:
                min_len = len(temp)
                ans = temp
            elif len(temp) == min_len and temp < ans:
                ans = temp
                
        return ans