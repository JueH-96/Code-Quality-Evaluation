class Solution:
    def minimumOperations(self, nums: List[int]) -> int:
        from collections import Counter
        
        # Count the frequency of each number in the array
        freq = Counter(nums)
        
        # Calculate the number of operations needed
        operations = 0
        while any(count > 1 for count in freq.values()):
            # Perform the operation: remove the first 3 elements
            operations += 1
            if len(nums) >= 3:
                # Remove the first 3 elements
                for i in range(3):
                    freq[nums[i]] -= 1
                    if freq[nums[i]] == 0:
                        del freq[nums[i]]
                nums = nums[3:]
            else:
                # If less than 3 elements, clear the array
                nums = []
                freq.clear()
        
        return operations