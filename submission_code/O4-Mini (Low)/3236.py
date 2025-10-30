from typing import List

class Solution:
    def missingInteger(self, nums: List[int]) -> int:
        # Compute the longest sequential prefix sum
        total = nums[0]
        last = nums[0]
        for i in range(1, len(nums)):
            if nums[i] == last + 1:
                total += nums[i]
                last = nums[i]
            else:
                break
        
        # Build a set for O(1) lookups
        seen = set(nums)
        
        # Find the smallest integer >= total that's not in nums
        x = total
        while x in seen:
            x += 1
        
        return x

# Example usage:
if __name__ == "__main__":
    sol = Solution()
    print(sol.missingInteger([1,2,3,2,5]))        # Output: 6
    print(sol.missingInteger([3,4,5,1,12,14,13]))  # Output: 15