class Solution:
    def countInterestingSubarrays(self, nums: List[int], modulo: int, k: int) -> int:
        n = len(nums)
        
        ans = 0
        cur = 0
        cnt = [0] * modulo
        cnt[0] = 1
        
        for num in nums:
            cur += num % modulo == k
            cur %= modulo
            ans += cnt[cur]
            cnt[cur + k] += 1
            cnt[cur + k] %= modulo
        
        return ans