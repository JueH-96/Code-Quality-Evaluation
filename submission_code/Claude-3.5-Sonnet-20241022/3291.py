class Solution:
    def canSortArray(self, nums: List[int]) -> bool:
        n = len(nums)
        
        # Helper function to count set bits
        def count_bits(num):
            count = 0
            while num:
                count += num & 1
                num >>= 1
            return count
        
        # Get set bits count for each number
        bits = [count_bits(num) for num in nums]
        
        # Keep trying to bubble sort until no swaps are possible
        while True:
            swapped = False
            
            for i in range(n-1):
                # If adjacent elements are out of order and have same bits
                if nums[i] > nums[i+1] and bits[i] == bits[i+1]:
                    nums[i], nums[i+1] = nums[i+1], nums[i]
                    swapped = True
            
            # If no swaps were made in this pass, break
            if not swapped:
                break
        
        # Check if array is sorted
        return all(nums[i] <= nums[i+1] for i in range(n-1))