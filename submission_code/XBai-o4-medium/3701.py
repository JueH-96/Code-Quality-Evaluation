class Solution:
    def minCostGoodCaption(self, caption: str) -> str:
        # Split into runs
        runs = []
        n = len(caption)
        if n == 0:
            return ""
        current_char = caption[0]
        count = 1
        for i in range(1, n):
            if caption[i] == current_char:
                count += 1
            else:
                runs.append((current_char, count))
                current_char = caption[i]
                count = 1
        runs.append((current_char, count))
        m = len(runs)
        
        # Compute prefix sums of run lengths
        prefix_sum = [0] * (m + 1)
        for i in range(m):
            prefix_sum[i+1] = prefix_sum[i] + runs[i][1]
        
        # Dynamic programming
        INF = float('inf')
        # dp[i] = (cost, prev_j, target_char)
        dp = [(INF, -1, None) for _ in range(m + 1)]
        dp[0] = (0, -1, None)
        
        for i in range(1, m + 1):
            # i is the number of runs processed. We need to find all a such that the group from a to i-1 is valid.
            # sum from a to i-1 is prefix_sum[i] - prefix_sum[a] >= 3
            for a in range(i):
                group_sum = prefix_sum[i] - prefix_sum[a]
                if group_sum >= 3:
                    # collect the chars and counts
                    chars = []
                    counts = []
                    for run_idx in range(a, i):
                        chars.append(runs[run_idx][0])
                        counts.append(runs[run_idx][1])
                    # compute the weighted median
                    sorted_chars = sorted([(chars[j], counts[j]) for j in range(len(chars))], key=lambda x: x[0])
                    total = group_sum
                    acc = 0
                    median_char = None
                    for (ch, cnt) in sorted_chars:
                        acc += cnt
                        if 2 * acc >= total:  # 2*acc >= total
                            median_char = ch
                            break
                    # compute the cost
                    cost = 0
                    for j in range(len(chars)):
                        original_char = chars[j]
                        cnt = counts[j]
                        delta = ord(median_char) - ord(original_char)
                        cost += cnt * abs(delta)
                    # update dp[i]
                    prev_j = a
                    current_cost = dp[prev_j][0] + cost
                    if current_cost < dp[i][0]:
                        dp[i] = (current_cost, prev_j, median_char)
                    elif current_cost == dp[i][0] and dp[i][0] != INF:
                        # compare current median_char with existing one.
                        existing_median = dp[i][2]
                        if median_char < existing_median:
                            dp[i] = (current_cost, prev_j, median_char)
        
        # After filling dp, check if dp[m] is valid.
        if dp[m][0] == INF:
            return ""
        
        # Reconstruct the string
        res = []
        current_i = m
        while current_i > 0:
            prev_j = dp[current_i][1]
            target_char = dp[current_i][2]
            # the group is from prev_j to current_i-1 (run indices)
            group_start = prev_j
            group_end = current_i - 1
            group_length = 0
            for idx in range(group_start, group_end + 1):
                group_length += runs[idx][1]
            res.append(target_char * group_length)
            current_i = prev_j
        
        res = ''.join(reversed(res))
        return res