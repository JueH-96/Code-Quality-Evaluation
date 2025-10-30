class Solution:
    def countKConstraintSubstrings(self, s: str, k: int) -> int:
        count = 0
        n = len(s)
        
        # Generate all possible substrings
        for i in range(n):
            for j in range(i + 1, n + 1):
                substring = s[i:j]
                
                # Count the number of 0's and 1's in the substring
                zeros = substring.count('0')
                ones = substring.count('1')
                
                # Check if the substring satisfies the k-constraint
                if zeros <= k or ones <= k:
                    count += 1
                    
        return count