class Solution:
    def findMinimumOperations(self, s1: str, s2: str, s3: str) -> int:
        # We can only delete characters from the right, so any final equal string
        # must be a non-empty prefix of s1, s2, and s3.
        # We will look for the longest common prefix among s1, s2, and s3 of length L>=1.
        # For each valid L, we check if s1[:L], s2[:L], and s3[:L] are all equal.
        # Among all valid L, we pick the one yielding the minimum total cost.
        
        # If no common prefix of length >= 1 exists, we return -1.
        
        min_len = min(len(s1), len(s2), len(s3))
        answer = float('inf')
        
        # Try prefix lengths from largest to smallest
        for length in range(min_len, 0, -1):
            if s1[:length] == s2[:length] == s3[:length]:
                # All share the same prefix of this length
                cost = (len(s1) - length) + (len(s2) - length) + (len(s3) - length)
                answer = min(answer, cost)
        
        return answer if answer != float('inf') else -1