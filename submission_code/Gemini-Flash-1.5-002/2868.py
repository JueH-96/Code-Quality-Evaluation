class Solution:
    def continuousSubarrays(self, nums: List[int]) -> int:
        count = 0
        n = len(nums)
        for i in range(n):
            for j in range(i, n):
                is_continuous = True
                sub_array = nums[i:j+1]
                for k in range(len(sub_array)):
                    for l in range(k + 1, len(sub_array)):
                        if abs(sub_array[k] - sub_array[l]) > 2:
                            is_continuous = False
                            break
                    if not is_continuous:
                        break
                if is_continuous:
                    count += 1
        return count