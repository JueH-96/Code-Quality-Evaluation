from typing import List

class Solution:
    def minimumRightShifts(self, nums: List[int]) -> int:
        n = len(nums)
        sorted_nums = sorted(nums)
        
        # If already sorted, no right shifts are needed.
        if nums == sorted_nums:
            return 0

        # For the array to be sortable by right shifts, it must be a rotation of the sorted array.
        # The smallest element in the sorted array (sorted_nums[0]) will be the pivot.
        pivot = nums.index(sorted_nums[0])
        
        # Verify that starting from the pivot, the array matches the sorted order.
        for i in range(n):
            if nums[(pivot + i) % n] != sorted_nums[i]:
                return -1
        
        # The minimum right shifts required is (n - pivot) % n.
        # (n - pivot) shifts bring the pivot element to index 0.
        return (n - pivot) % n

# Example usage:
if __name__ == "__main__":
    sol = Solution()
    print(sol.minimumRightShifts([3, 4, 5, 1, 2]))  # Expected output: 2
    print(sol.minimumRightShifts([1, 3, 5]))        # Expected output: 0
    print(sol.minimumRightShifts([2, 1, 4]))        # Expected output: -1