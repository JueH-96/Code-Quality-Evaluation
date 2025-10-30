class Solution:
    def hasIncreasingSubarrays(self, nums: List[int], k: int) -> bool:
        def is_strictly_increasing(arr, start, length):
            for i in range(start + 1, start + length):
                if arr[i] <= arr[i - 1]:
                    return False
            return True
        
        # Check all possible starting positions for the first subarray
        for i in range(len(nums) - 2 * k + 1):
            # Check if subarray starting at i is strictly increasing
            if is_strictly_increasing(nums, i, k):
                # Check if adjacent subarray starting at i + k is strictly increasing
                if is_strictly_increasing(nums, i + k, k):
                    return True
        
        return False