class Solution:
    def findMinimumOperations(self, s1: str, s2: str, s3: str) -> int:
        # We can only remove characters from the right, 
        # and we are not allowed to operate on strings of length 1.
        # So the final string must be of length at least 1.
        # The only possible final strings, for each original string,
        # are its prefixes. So, if we can make the strings equal,
        # they must equal some prefix of length k (with k >= 1).
        #
        # To minimize the number of removals (operations),
        # choose the longest possible common prefix among s1, s2, s3.
        
        # Determine the maximum possible candidate length (the smallest length)
        min_len = min(len(s1), len(s2), len(s3))
        best = -1
        
        # Check from longest possible (min_len) down to 1
        # because longer common prefix => fewer removals.
        for k in range(min_len, 0, -1):
            # Check if s1, s2, s3 share the same prefix of length k.
            if s1[:k] == s2[:k] and s2[:k] == s3[:k]:
                best = k
                break
        
        # If no common prefix of positive length exists, then it's not possible.
        if best == -1:
            return -1
        
        # Count operations: For each string, we remove characters from
        # the right until its length equals best.
        operations = (len(s1) - best) + (len(s2) - best) + (len(s3) - best)
        return operations

# Example usage:
solution = Solution()
print(solution.findMinimumOperations("abc", "abb", "ab"))  # Expected output: 2
print(solution.findMinimumOperations("dac", "bac", "cac"))  # Expected output: -1