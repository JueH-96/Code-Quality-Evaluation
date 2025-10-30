class Solution:
    def minLength(self, s: str, numOps: int) -> int:
        n = len(s)
        
        # Initialize the minimum length to the length of the string
        min_length = n
        
        # Iterate through all possible starting indices
        for i in range(n):
            # Initialize the current length to 0
            curr_length = 0
            
            # Initialize the number of operations used to 0
            ops_used = 0
            
            # Iterate through the string starting from the current index
            for j in range(i, n):
                # Increment the current length
                curr_length += 1
                
                # If the current character is different from the previous one,
                # increment the number of operations used
                if j > i and s[j] != s[j-1]:
                    ops_used += 1
                
                # If the number of operations used is greater than the allowed number of operations,
                # break out of the inner loop
                if ops_used > numOps:
                    break
                
                # Update the minimum length if the current length is smaller
                min_length = min(min_length, curr_length)
        
        return min_length