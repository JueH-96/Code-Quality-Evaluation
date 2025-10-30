class Solution:
    def longestSubsequence(self, nums: List[int]) -> int:
        n = len(nums)
        ans = 1
        suf_list = [[0] * 301 for _ in range(n)]
        
        for i in range(n):
            last_state_i = [0] * 301
            for j in range(i):
                d_val = abs(nums[i] - nums[j])
                candidate_val = max(2, suf_list[j][d_val] + 1)
                if candidate_val > last_state_i[d_val]:
                    last_state_i[d_val] = candidate_val
            suf_i = [0] * 301
            suf_i[300] = last_state_i[300]
            for d in range(299, -1, -1):
                suf_i[d] = max(last_state_i[d], suf_i[d+1])
            suf_list[i] = suf_i
            current_max = max(last_state_i)
            if current_max > ans:
                ans = current_max
        return ans