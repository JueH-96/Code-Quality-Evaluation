class Solution:
    def hasIncreasingSubarrays(self, nums: List[int], k: int) -> bool:
        valid_starts = []
        for i in range(len(nums) - k + 1):
            is_increasing = True
            for j in range(i, i + k - 1):
                if nums[j] >= nums[j + 1]:
                    is_increasing = False
                    break
            if is_increasing:
                valid_starts.append(i)
        valid_set = set(valid_starts)
        for i in valid_set:
            if (i + k) in valid_set:
                return True
        return False