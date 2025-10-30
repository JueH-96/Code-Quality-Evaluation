class Solution:
    def canSortArray(self, nums: List[int]) -> bool:
        # Handle edge case of single element
        if len(nums) == 1:
            return True
        
        # Function to count set bits
        def count_set_bits(x):
            return bin(x).count('1')
        
        # Precompute set bit counts for all elements
        set_bits = [count_set_bits(x) for x in nums]
        
        # Make a copy of the array and sort it
        sorted_nums = sorted(nums)
        
        # Work on a copy of the original array
        arr = nums.copy()
        
        # Perform swaps until no more can be made
        while True:
            swapped = False
            for i in range(len(arr) - 1):
                if set_bits[i] == set_bits[i + 1] and arr[i] > arr[i + 1]:
                    # Swap the elements
                    arr[i], arr[i + 1] = arr[i + 1], arr[i]
                    swapped = True
            if not swapped:
                break
        
        # After all possible swaps, check if the array is sorted
        return arr == sorted_nums