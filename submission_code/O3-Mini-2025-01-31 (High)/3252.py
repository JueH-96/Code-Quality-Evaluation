from typing import List

class Solution:
    def incremovableSubarrayCount(self, nums: List[int]) -> int:
        n = len(nums)
        
        def is_strictly_increasing(seq: List[int]) -> bool:
            # An empty sequence or a single element sequence is strictly increasing.
            for i in range(1, len(seq)):
                if seq[i] <= seq[i-1]:
                    return False
            return True
        
        count = 0
        # Iterate over all non-empty contiguous subarrays (by their start and end indices)
        for i in range(n):
            for j in range(i, n):
                # Remove the subarray nums[i:j+1]
                new_seq = nums[:i] + nums[j+1:]
                if is_strictly_increasing(new_seq):
                    count += 1
        return count

# The following code is for local testing if needed.
if __name__ == "__main__":
    solution = Solution()
    print(solution.incremovableSubarrayCount([1, 2, 3, 4]))  # Expected output: 10
    print(solution.incremovableSubarrayCount([6, 5, 7, 8]))  # Expected output: 7
    print(solution.incremovableSubarrayCount([8, 7, 6, 6]))  # Expected output: 3