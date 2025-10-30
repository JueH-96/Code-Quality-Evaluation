class Solution:
    def minimumString(self, a: str, b: str, c: str) -> str:
        def merge_two_strings(s1, s2):
            """Find the shortest string that has both s1 and s2 as substrings."""
            # If one string contains the other, return the containing string
            if s1 in s2:
                return s2
            if s2 in s1:
                return s1
            
            candidates = []
            
            # Try simple concatenation in both orders
            candidates.append(s1 + s2)
            candidates.append(s2 + s1)
            
            # Try to find overlap where end of s1 matches beginning of s2
            for i in range(1, min(len(s1), len(s2)) + 1):
                if s1[-i:] == s2[:i]:
                    candidate = s1 + s2[i:]
                    if s1 in candidate and s2 in candidate:  # Verify it's a valid superstring
                        candidates.append(candidate)
            
            # Try to find overlap where end of s2 matches beginning of s1
            for i in range(1, min(len(s1), len(s2)) + 1):
                if s2[-i:] == s1[:i]:
                    candidate = s2 + s1[i:]
                    if s1 in candidate and s2 in candidate:  # Verify it's a valid superstring
                        candidates.append(candidate)
            
            # Sort by length, then lexicographically
            candidates.sort(key=lambda x: (len(x), x))
            return candidates[0]
        
        # Consider all possible orderings of the three strings
        orders = [(a, b, c), (a, c, b), (b, a, c), (b, c, a), (c, a, b), (c, b, a)]
        
        min_length = float('inf')
        min_string = ""
        
        for order in orders:
            s1, s2, s3 = order
            # Merge the first two strings, then merge the result with the third
            merged = merge_two_strings(merge_two_strings(s1, s2), s3)
            
            # Update answer if we found a shorter string or a lexicographically smaller one of same length
            if len(merged) < min_length or (len(merged) == min_length and merged < min_string):
                min_length = len(merged)
                min_string = merged
        
        return min_string