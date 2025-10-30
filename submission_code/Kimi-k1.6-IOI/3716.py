class Solution:
    def longestSubsequence(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 0:
            return 0
        max_len = 1
        suffix_max_list = []
        for i in range(n):
            diff_len = [0] * 300
            for j in range(i):
                d_new = abs(nums[i] - nums[j])
                max_prev = suffix_max_list[j][d_new] if j < len(suffix_max_list) else 0
                if max_prev == 0:
                    candidate = 2
                else:
                    candidate = max_prev + 1
                if candidate > diff_len[d_new]:
                    diff_len[d_new] = candidate
            # Compute suffix_max for current diff_len
            suffix_max = [0] * 300
            if i > 0:
                suffix_max[299] = diff_len[299]
                for d in range(298, -1, -1):
                    suffix_max[d] = max(diff_len[d], suffix_max[d + 1])
            else:
                suffix_max = [0] * 300
            suffix_max_list.append(suffix_max)
            current_max = suffix_max[0] if suffix_max else 0
            if current_max > max_len:
                max_len = current_max
        return max_len