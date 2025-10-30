class Solution:
    def minimumString(self, a: str, b: str, c: str) -> str:
        # Helper function: merge two strings with maximal overlap
        def merge(s1, s2):
            # if s2 is already a substring of s1, return s1
            if s2 in s1:
                return s1
            # Check the maximum possible overlap where end of s1 overlap with beginning of s2
            max_overlap = 0
            for i in range(1, min(len(s1), len(s2)) + 1):
                if s1[-i:] == s2[:i]:
                    max_overlap = i
            return s1 + s2[max_overlap:]
        
        # Remove any string that is a substring of any other one
        strs = [a, b, c]
        unique = []
        for s in strs:
            if any(s != t and s in t for t in strs):
                continue
            unique.append(s)
        
        # If after removal, nothing to merge, simply return one
        if not unique:
            return ""
        
        from itertools import permutations
        
        best = None
        # Try all orderings from the unique list
        for order in permutations(unique):
            merged = order[0]
            for s in order[1:]:
                merged = merge(merged, s)
            # Update best candidate:
            if best is None:
                best = merged
            else:
                if len(merged) < len(best) or (len(merged) == len(best) and merged < best):
                    best = merged
        return best
                        
# Example usage:
# sol = Solution()
# print(sol.minimumString("abc", "bca", "aaa"))  # Expected: "aaabca"
# print(sol.minimumString("ab", "ba", "aba"))  # Expected: "aba"