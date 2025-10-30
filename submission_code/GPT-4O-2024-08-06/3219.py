class Solution:
    def lexicographicallySmallestArray(self, nums: List[int], limit: int) -> List[int]:
        n = len(nums)
        for i in range(n):
            # Try to find the smallest element that can be swapped with nums[i]
            min_index = i
            for j in range(i + 1, n):
                if abs(nums[i] - nums[j]) <= limit and nums[j] < nums[min_index]:
                    min_index = j
            # Swap if we found a smaller element
            if min_index != i:
                nums[i], nums[min_index] = nums[min_index], nums[i]
        return nums