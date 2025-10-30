import math

class Solution:
    """
    Calculates the number of substrings with dominant ones.
    A string has dominant ones if the number of ones is greater than or equal to 
    the square of the number of zeros.
    """
    def numberOfSubstrings(self, s: str) -> int:
        """
        Counts substrings with dominant ones using an optimized approach.
        The time complexity is O(N*sqrt(N)) in the worst case, where N is the length of s.
        The optimization involves two main ideas:
        1. Early termination: If the number of zeros `z` in a substring s[i..j] exceeds sqrt(N), 
           then `z*z > N`. Since the number of ones `o` is at most N, `o < z*z`. 
           The condition `o >= z*z` cannot be met for this substring or any extension s[i..k] where k > j.
           Thus, we can break the inner loop early.
        2. Jump logic: If a substring s[i..j] satisfies the condition `o >= z*z`, then any extension 
           s[i..p] where s[j+1..p] consists only of '1's will also satisfy the condition. This is because 
           `zeros` count remains the same while `ones` count increases. We can calculate the number of such 
           extensions quickly using precomputed next zero positions and jump the index `j` forward.

        Args:
            s: The binary string.
        Returns:
            The total number of substrings with dominant ones.
        """
        n = len(s)
        count = 0
        
        # Precompute the index of the next zero for each position i.
        # next_zero[i] stores the index of the first '0' at or after index i.
        # If no zero exists at or after i, next_zero[i] = n.
        # We use size n+1 for easier boundary handling (next_zero[n] = n).
        next_zero = [n] * (n + 1)  
        for i in range(n - 1, -1, -1):
            if s[i] == '0':
                next_zero[i] = i
            else:
                # If s[i] is '1', the next zero is the same as the next zero for i+1
                next_zero[i] = next_zero[i+1]

        # Maximum number of zeros relevant for the condition ones >= zeros*zeros.
        # Calculated as floor(sqrt(n)).
        max_z_limit = int(math.sqrt(n)) 

        # Iterate through all possible start indices `i` of substrings
        for i in range(n):
            zeros = 0
            ones = 0 
            
            # Iterate through end indices `j` using a while loop.
            # The while loop is necessary because we modify `j` inside the loop for the jump logic.
            j = i 
            while j < n:
                # Update counts based on the character at index j
                if s[j] == '0':
                    zeros += 1
                else:
                    ones += 1
                
                # Optimization: Early termination if zeros exceed sqrt(N)
                if zeros > max_z_limit:
                     break # Exit the inner loop (while j < n) for this starting index i

                # Check if the current substring s[i..j] has dominant ones
                if ones >= zeros * zeros:
                    # If the condition is met for s[i..j], it also holds for extensions
                    # s[i..p] as long as s[j+1..p] contains only '1's.
                    
                    # Find the index k of the next zero occurring after index j.
                    # This marks the end of the consecutive block of '1's following index j.
                    k = next_zero[j+1] 
                    
                    # All substrings s[i..p] for p from j to k-1 are valid because they have the
                    # same number of zeros as s[i..j] but potentially more ones.
                    # There are (k-1 - j + 1) = k - j such substrings. Add this to the total count.
                    count += (k - j)
                    
                    # To continue the loop efficiently after finding a valid block,
                    # update the 'ones' count to reflect the inclusion of '1's up to index k-1.
                    # The number of ones added corresponds to the characters s[j+1]...s[k-1].
                    num_ones_added = k - 1 - j
                    # This check handles the case k = j+1, where there are no intermediate '1's.
                    if num_ones_added > 0: 
                      ones += num_ones_added
                    
                    # Advance j to k-1. The outer while loop condition check and the final `j += 1` 
                    # will correctly position j to k for the next iteration, effectively skipping 
                    # the processed block of '1's and potentially starting the check from the next zero position.
                    j = k - 1 
                
                # Increment j to consider the next character / substring end point.
                j += 1
        
        return count