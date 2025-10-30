class Solution:
    def countKConstraintSubstrings(self, s: str, k: int) -> int:
        n = len(s)
        count = 0
        
        # Helper function to check if substring satisfies k-constraint
        def satisfies_k_constraint(sub):
            zeros = sub.count('0')
            ones = sub.count('1')
            return zeros <= k or ones <= k
        
        # Check all possible substrings
        for start in range(n):
            for end in range(start + 1, n + 1):
                if satisfies_k_constraint(s[start:end]):
                    count += 1
        
        return count