from collections import Counter

class Solution:
    def minOperations(self, nums: List[int]) -> int:
        # Count the frequency of each element in the array
        freq = Counter(nums)
        
        # Initialize the number of operations to 0
        operations = 0
        
        # Iterate through the frequency dictionary
        while freq:
            # Find the element with the highest frequency
            max_freq = max(freq.values())
            
            # If the maximum frequency is less than 3, it's impossible to empty the array
            if max_freq < 3:
                return -1
            
            # Decrement the frequency of the element with the highest frequency
            for num, count in freq.items():
                if count == max_freq:
                    freq[num] -= 3
                    if freq[num] <= 0:
                        del freq[num]
                    break
            
            # Increment the number of operations
            operations += 1
        
        return operations