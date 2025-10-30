class Solution:
    def continuousSubarrays(self, nums: List[int]) -> int:
        n = len(nums)
        count = 0
        for i in range(n):
            for j in range(i, n):
                is_continuous = True
                for k1 in range(i, j + 1):
                    for k2 in range(i, j + 1):
                        if abs(nums[k1] - nums[k2]) > 2:
                            is_continuous = False
                            break
                    if not is_continuous:
                        break
                if is_continuous:
                    count += 1
        return count