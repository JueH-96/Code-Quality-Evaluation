class Solution:
    def minimumValueSum(self, nums: List[int], andValues: List[int]) -> int:
        n = len(nums)
        m = len(andValues)
        if m > n:
            return -1
        INF = float('inf')
        dp = [[INF] * (n + 1) for _ in range(m + 1)]
        dp[0][0] = 0
        
        from collections import defaultdict

        # For each j, we maintain a list of dictionaries tracking the current_and and the earliest start
        # and_dicts[j][i] is a dictionary {current_and: (start, min_sum)}
        and_dicts = [ [defaultdict(lambda: (INF, INF)) for _ in range(n + 1)] for _ in range(m + 1) ]
        and_dicts[0][0] = {0: (0, 0)}  # j=0, sum=0, start=0

        for j in range(1, m + 1):
            current_target = andValues[j - 1]
            current_dict = defaultdict(lambda: (INF, INF))
            for i in range(1, n + 1):
                remaining = n - i
                if remaining < (m - j):
                    and_dicts[j][i] = current_dict.copy()
                    continue

                if j == 1:
                    # For j=1, the previous is j=0, start is 0
                    new_and = nums[0] if i == 1 else (and_dicts[j][i-1][0][0] & nums[i-1] if and_dicts[j][i-1] else nums[i-1])
                    sum_so_far = dp[j-1][0] + nums[i-1]
                    current_dict[new_and] = (0, sum_so_far)
                else:
                    prev_dict = and_dicts[j-1][i-1]
                    temp_dict = defaultdict(lambda: (INF, INF))
                    for prev_and in prev_dict:
                        prev_start, prev_sum = prev_dict[prev_and]
                        new_and = prev_and & nums[i-1]
                        if prev_start < temp_dict[new_and][0] or (prev_start == temp_dict[new_and][0] and prev_sum < temp_dict[new_and][1]):
                            temp_dict[new_and] = (prev_start, prev_sum)
                    # Update current_dict with temp_dict
                    current_dict = defaultdict(lambda: (INF, INF))
                    for key in temp_dict:
                        current_dict[key] = temp_dict[key]
                    # Check if current_target is in current_dict
                    if current_target in current_dict:
                        start, sum_so_far = current_dict[current_target]
                        sum_candidate = sum_so_far + nums[i-1]
                        if sum_candidate < dp[j][i]:
                            dp[j][i] = sum_candidate
                and_dicts[j][i] = current_dict.copy()

        return dp[m][n] if dp[m][n] != INF else -1