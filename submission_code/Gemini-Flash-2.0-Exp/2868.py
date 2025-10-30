class Solution:
    def continuousSubarrays(self, nums: List[int]) -> int:
        n = len(nums)
        count = 0
        for i in range(n):
            for j in range(i, n):
                subarray = nums[i:j+1]
                is_continuous = True
                for k in range(len(subarray)):
                    for l in range(k + 1, len(subarray)):
                        if abs(subarray[k] - subarray[l]) > 2:
                            is_continuous = False
                            break
                    if not is_continuous:
                        break
                if is_continuous:
                    count += 1
        return count