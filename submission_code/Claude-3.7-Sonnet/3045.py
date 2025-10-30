class Solution:
    def minimumRightShifts(self, nums: List[int]) -> int:
        n = len(nums)
        drop_count = 0
        drop_index = -1
        
        # Count the number of drops and track the last drop's index
        for i in range(n):
            if nums[i] > nums[(i + 1) % n]:
                drop_count += 1
                drop_index = i
        
        # If there's more than one drop, it's not sortable by right shifts
        if drop_count > 1:
            return -1
        # If there's no drop, it's already sorted
        if drop_count == 0:
            return 0
        
        # If there's exactly one drop, we need to right shift to bring the smallest element to the beginning
        # The smallest element is at index (drop_index + 1) % n
        # To bring it to the beginning, we need (n - ((drop_index + 1) % n)) % n right shifts
        return (n - ((drop_index + 1) % n)) % n