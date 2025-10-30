from typing import List

class Solution:
    def incremovableSubarrayCount(self, nums: List[int]) -> int:
        n = len(nums)
        # Compute prefix_valid[k] = True if the subarray nums[0:k] is strictly increasing.
        prefix_valid = [True] * (n + 1)
        prefix_valid[0] = True  # empty is strictly increasing.
        for k in range(1, n + 1):
            if k == 1:
                prefix_valid[k] = True
            else:
                prefix_valid[k] = prefix_valid[k - 1] and (nums[k - 2] < nums[k - 1])
        
        # Compute suffix_valid[k] = True if the subarray nums[k:n] is strictly increasing.
        suffix_valid = [True] * (n + 1)
        suffix_valid[n] = True  # empty is strictly increasing.
        for k in range(n - 1, -1, -1):
            if k == n - 1:
                suffix_valid[k] = True
            else:
                suffix_valid[k] = suffix_valid[k + 1] and (nums[k] < nums[k + 1])
        
        count = 0
        # Consider every subarray to remove, defined by indices i and j (inclusive)
        # The remaining array becomes: nums[0:i] + nums[j+1:n]
        for i in range(0, n):
            for j in range(i, n):
                # Check that the left part is strictly increasing.
                if not prefix_valid[i]:
                    continue
                # Check that the right part is strictly increasing.
                if not suffix_valid[j + 1]:
                    continue
                # If both left and right parts exist, ensure their boundary elements satisfy:
                # last element of left < first element of right.
                if i > 0 and j + 1 < n:
                    if not (nums[i - 1] < nums[j + 1]):
                        continue
                count += 1
        return count

# Simple tests:
if __name__ == '__main__':
    sol = Solution()
    # Example 1: expected output 10
    print(sol.incremovableSubarrayCount([1,2,3,4]))
    # Example 2: expected output 7
    print(sol.incremovableSubarrayCount([6,5,7,8]))
    # Example 3: expected output 3
    print(sol.incremovableSubarrayCount([8,7,6,6]))