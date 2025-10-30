from typing import List

class Solution:
    def getFinalState(self, nums: List[int], k: int, multiplier: int) -> List[int]:
        # Perform k operations
        for _ in range(k):
            # Find the minimum value and its first occurrence index
            min_val = min(nums)
            idx = nums.index(min_val)
            # Replace it with its multiplied value
            nums[idx] = nums[idx] * multiplier
        return nums

# Example usage:
if __name__ == "__main__":
    sol = Solution()
    print(sol.getFinalState([2,1,3,5,6], 5, 2))  # Output: [8,4,6,5,6]
    print(sol.getFinalState([1,2], 3, 4))        # Output: [16,8]