class Solution:
    def minChanges(self, nums: List[int], k: int) -> int:
        n = len(nums)
        ans = float('inf')
        for x in range(k + 1):
            count = 0
            for i in range(n // 2):
                a = nums[i]
                b = nums[n - 1 - i]
                if abs(a - b) != x:
                    if abs(a - b) + min(abs(a - (b + x)), abs(a - (b - x))) <= 2 * k:
                        count +=1
                    else:
                        count +=2

            ans = min(ans, count)
        return ans