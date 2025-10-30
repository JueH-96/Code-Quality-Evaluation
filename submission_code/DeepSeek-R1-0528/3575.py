class Solution:
    def maxValue(self, nums: List[int], k: int) -> int:
        n = len(nums)
        prefix_set = [set() for _ in range(n)]
        dp_pre = [set() for _ in range(k+1)]
        dp_pre[0].add(0)
        
        for i in range(n):
            new_dp_pre = [set() for _ in range(k+1)]
            for j in range(k+1):
                new_dp_pre[j] = set(dp_pre[j])
            for j in range(k):
                for or_val in dp_pre[j]:
                    new_or = or_val | nums[i]
                    new_dp_pre[j+1].add(new_or)
            dp_pre = new_dp_pre
            prefix_set[i] = dp_pre[k]
        
        suffix_set = [set() for _ in range(n)]
        dp_suf = [set() for _ in range(k+1)]
        dp_suf[0].add(0)
        
        for i in range(n-1, -1, -1):
            new_dp_suf = [set() for _ in range(k+1)]
            for j in range(k+1):
                new_dp_suf[j] = set(dp_suf[j])
            for j in range(k):
                for or_val in dp_suf[j]:
                    new_or = or_val | nums[i]
                    new_dp_suf[j+1].add(new_or)
            dp_suf = new_dp_suf
            suffix_set[i] = dp_suf[k]
        
        ans = 0
        for m in range(k, n - k + 1):
            for a in prefix_set[m-1]:
                for b in suffix_set[m]:
                    ans = max(ans, a ^ b)
        return ans