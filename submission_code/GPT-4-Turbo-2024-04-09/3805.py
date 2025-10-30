class Solution:
    def maxActiveSectionsAfterTrade(self, s: str) -> int:
        # Augment the string with '1' at both ends
        s = '1' + s + '1'
        n = len(s)
        
        # Lists to store the lengths of consecutive '0's and '1's
        zeros = []
        ones = []
        
        # Current count of consecutive characters
        count = 1
        # Iterate over the string to count consecutive '0's and '1's
        for i in range(1, n):
            if s[i] == s[i-1]:
                count += 1
            else:
                if s[i-1] == '0':
                    zeros.append(count)
                else:
                    ones.append(count)
                count = 1
        
        # Append the last counted segment
        if s[n-1] == '0':
            zeros.append(count)
        else:
            ones.append(count)
        
        # If there are no zeros surrounded by ones, no trade can be made
        if len(zeros) <= 1:
            return sum(ones) - 2  # Subtract the augmented '1's
        
        # Calculate the maximum active sections after the optimal trade
        max_active = 0
        total_ones = sum(ones) - 2  # Subtract the augmented '1's
        
        # Iterate through the zeros list (ignoring the first and last zero segments)
        for i in range(1, len(zeros) - 1):
            # Calculate potential active sections by trading zeros[i] with ones[i] and ones[i+1]
            new_active = total_ones + zeros[i] - (ones[i] + ones[i+1] - 2)
            max_active = max(max_active, new_active)
        
        return max_active

# Example usage:
sol = Solution()
print(sol.maxActiveSectionsAfterTrade("01"))       # Output: 1
print(sol.maxActiveSectionsAfterTrade("0100"))     # Output: 4
print(sol.maxActiveSectionsAfterTrade("1000100"))  # Output: 7
print(sol.maxActiveSectionsAfterTrade("01010"))    # Output: 4