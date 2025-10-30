class Solution:
    def canSortArray(self, nums: List[int]) -> bool:
        def count_set_bits(n):
            return bin(n).count('1')
        
        n = len(nums)
        if n <= 1:
            return True
        
        # Create a copy to work with
        arr = nums[:]
        
        # Process segments of elements with same set bit count
        i = 0
        while i < n:
            j = i
            current_bits = count_set_bits(arr[i])
            
            # Find the end of the current segment
            while j < n and count_set_bits(arr[j]) == current_bits:
                j += 1
            
            # Sort the segment
            arr[i:j] = sorted(arr[i:j])
            i = j
        
        # Check if the result is sorted
        return arr == sorted(nums)