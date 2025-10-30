import bisect

class Solution:
    def maxSubarraySum(self, nums: list[int], k: int) -> int:
        n = len(nums)
        prefix_sum = [0] * (n + 1)
        for i in range(1, n + 1):
            prefix_sum[i] = prefix_sum[i-1] + nums[i-1]
        
        # Create groups based on j % k
        groups = [[] for _ in range(k)]
        for j in range(n + 1):
            r = j % k
            groups[r].append(j)
        
        # Precompute suffix max for each group
        suffix_max_list = []
        for r in range(k):
            group = groups[r]
            m = len(group)
            if m == 0:
                suffix_max_list.append([])
                continue
            suffix_max = [0] * m
            suffix_max[-1] = prefix_sum[group[-1]]
            for i in range(m - 2, -1, -1):
                suffix_max[i] = max(prefix_sum[group[i]], suffix_max[i + 1])
            suffix_max_list.append(suffix_max)
        
        max_sum = float('-inf')
        for a in range(n + 1):
            r = a % k
            group = groups[r]
            suffix_max = suffix_max_list[r]
            pos = bisect.bisect_left(group, a)
            # Ensure a is found in the group (which it should be)
            if pos < len(group) and group[pos] == a:
                if pos + 1 < len(group):
                    current_sum = suffix_max[pos + 1] - prefix_sum[a]
                    if current_sum > max_sum:
                        max_sum = current_sum
        
        return max_sum