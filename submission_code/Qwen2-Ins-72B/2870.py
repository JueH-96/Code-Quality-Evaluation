class Solution:
    def alternatingSubarray(self, nums: List[int]) -> int:
        n = len(nums)
        ans = -1
        for i in range(n):
            for j in range(i+1, n):
                if nums[j-1] + 1 == nums[j]:
                    k = j
                    while k < n and (nums[k-1] + 1 == nums[k] if (k-i) % 2 == 1 else nums[k-1] == nums[k]):
                        k += 1
                    ans = max(ans, k-i)
                    break
        return ans