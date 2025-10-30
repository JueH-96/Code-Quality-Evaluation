class Solution:
    def continuousSubarrays(self, nums: List[int]) -> int:
        n = len(nums)
        ans = 0
        for i in range(n):
            for j in range(i, n):
                sub_array = nums[i:j+1]
                valid = True
                for k1 in range(len(sub_array)):
                    for k2 in range(len(sub_array)):
                        if abs(sub_array[k1] - sub_array[k2]) > 2:
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    ans += 1
        return ans