class Solution:
    def checkArray(self, nums: List[int], k: int) -> bool:
        n = len(nums)
        current_nums = list(nums)
        for i in range(n):
            if current_nums[i] < 0:
                return False
            if current_nums[i] > 0:
                needed_reduction = current_nums[i]
                if i + k > n:
                    return False
                for j in range(i, i + k):
                    current_nums[j] -= needed_reduction
        for val in current_nums:
            if val != 0:
                return False
        return True