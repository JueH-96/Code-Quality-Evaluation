class Solution:
    def countKConstraintSubstrings(self, s: str, k: int) -> int:
        count = 0
        n = len(s)
        
        # Iterate over all possible starting positions (i) of a substring
        for i in range(n):
            current_zeros = 0
            current_ones = 0
            
            # Iterate over all possible ending positions (j) for the current starting position i.
            # This implicitly generates all substrings s[i...j].
            for j in range(i, n):
                # Update the counts of '0's and '1's for the current substring s[i:j+1]
                if s[j] == '0':
                    current_zeros += 1
                else: # s[j] == '1'
                    current_ones += 1
                
                # Check if the current substring satisfies the k-constraint:
                # Either the number of 0's is at most k, OR the number of 1's is at most k.
                if current_zeros <= k or current_ones <= k:
                    count += 1
        
        return count