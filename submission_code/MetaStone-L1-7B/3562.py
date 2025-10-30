import bisect

class Solution:
    def maximumWeight(self, intervals: List[List[int]]) -> List[int]:
        intervals.sort(key=lambda x: x[1])
        n = len(intervals)
        dp = [{} for _ in range(n)]
        for k in range(1, 5):
            dp[k] = {'sum': 0, 'indices': []}
        
        for i in range(n):
            l_i, r_i, w_i = intervals[i]
            j = bisect.bisect_left(intervals, (l_i, float('inf')), 0, i)
            j -= 1
            
            new_dp = [{} for _ in range(5)]
            for k in range(1, 5):
                option1 = dp[k].get('sum', 0)
                option2_sum = 0
                option2_indices = []
                if j >= 0:
                    prev_k = k - 1
                    if prev_k >= 0:
                        if k - 1 <= 4:
                            if prev_k in range(5):
                                option2_sum = dp[j][prev_k]['sum'] + w_i
                                option2_indices = dp[j][prev_k]['indices'].copy()
                                option2_indices.append(i)
                            else:
                                option2_sum = 0
                                option2_indices = [i]
                new_dp[k] = {'sum': max(option1, option2_sum), 'indices': option2_indices.copy()}
            
            for k in range(5):
                if k < len(new_dp):
                    dp[i][k] = new_dp[k]
                else:
                    dp[i][k] = new_dp[k-1]
        
        max_sum = -1
        candidates = []
        for k in range(1, 5):
            if k >= len(dp):
                continue
            if 'sum' in dp[-1][k]:
                current_sum = dp[-1][k]['sum']
                if current_sum > max_sum:
                    max_sum = current_sum
                    candidates = [dp[-1][k]['indices']]
                elif current_sum == max_sum:
                    candidates.append(dp[-1][k]['indices'])
        
        if not candidates:
            return []
        
        min_indices = None
        for candidate in candidates:
            candidate_sorted = sorted(candidate)
            if min_indices is None:
                min_indices = candidate_sorted
            else:
                for a in range(len(min_indices)):
                    if min_indices[a] < candidate_sorted[a]:
                        min_indices = candidate_sorted
                        break
                    elif min_indices[a] > candidate_sorted[a]:
                        break
                    else:
                        continue
                else:
                    min_indices = min_indices
        
        return min_indices