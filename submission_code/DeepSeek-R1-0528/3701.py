import sys
sys.setrecursionlimit(1000000)

class Solution:
    def minCostGoodCaption(self, caption: str) -> str:
        n = len(caption)
        if n < 3:
            return ""
        
        INF = 10**18
        a_int = ord('a')
        letters = [chr(a_int + i) for i in range(26)]
        P = {}
        for idx, char in enumerate(letters):
            arr = [abs(ord(caption[0]) - a_int - idx)]
            for i in range(1, n):
                arr.append(arr[-1] + abs(ord(caption[i]) - a_int - idx))
            P[idx] = arr
        
        dp = [[INF] * 26 for _ in range(n)]
        from_start = [[-1] * 26 for _ in range(n)]
        from_prev_letter = [[-1] * 26 for _ in range(n)]
        
        pending_val_value_arr = [[INF] * 26 for _ in range(n)]
        pending_val_letter_arr = [[-1] * 26 for _ in range(n)]
        
        best_value = [INF] * 26
        best_j_for_c = [-1] * 26
        best_prev_letter = [-1] * 26
        
        for i in range(n):
            if i >= 3:
                j = i - 3
                for c in range(26):
                    if pending_val_value_arr[j][c] < best_value[c]:
                        best_value[c] = pending_val_value_arr[j][c]
                        best_j_for_c[c] = j
                        best_prev_letter[c] = pending_val_letter_arr[j][c]
            
            for c in range(26):
                if i >= 2:
                    cost_block = P[c][i]
                    if cost_block < dp[i][c]:
                        dp[i][c] = cost_block
                        from_start[i][c] = 0
                        from_prev_letter[i][c] = -1
                if i >= 3:
                    if best_value[c] < INF and best_value[c] + P[c][i] < dp[i][c]:
                        dp[i][c] = best_value[c] + P[c][i]
                        from_start[i][c] = best_j_for_c[c] + 1
                        from_prev_letter[i][c] = best_prev_letter[c]
            
            min_dp_i = min(dp[i])
            min_letter_set = set()
            for c in range(26):
                if dp[i][c] == min_dp_i:
                    min_letter_set.add(c)
            second_min_dp_i = INF
            for c in range(26):
                if dp[i][c] > min_dp_i and dp[i][c] < second_min_dp_i:
                    second_min_dp_i = dp[i][c]
            if second_min_dp_i == INF:
                second_min_dp_i = INF
            
            for c in range(26):
                if min_dp_i == INF:
                    pending_val_value = INF
                    best_prev = -1
                else:
                    if min_letter_set - {c}:
                        pending_val_value = min_dp_i - P[c][i]
                        best_prev = None
                        for x in range(26):
                            if x != c and dp[i][x] == min_dp_i:
                                if best_prev is None or x < best_prev:
                                    best_prev = x
                    else:
                        if second_min_dp_i == INF:
                            pending_val_value = INF
                            best_prev = -1
                        else:
                            pending_val_value = second_min_dp_i - P[c][i]
                            best_prev = None
                            for x in range(26):
                                if x != c and dp[i][x] == second_min_dp_i:
                                    if best_prev is None or x < best_prev:
                                        best_prev = x
                pending_val_value_arr[i][c] = pending_val_value
                pending_val_letter_arr[i][c] = best_prev if best_prev is not None else -1
        
        min_cost = min(dp[n-1])
        if min_cost == INF:
            return ""
        
        candidates = []
        for c in range(26):
            if dp[n-1][c] == min_cost:
                blocks = []
                cur_i = n - 1
                cur_c = c
                while cur_i >= 0:
                    s_index = from_start[cur_i][cur_c]
                    if s_index == -1:
                        break
                    letter_char = chr(a_int + cur_c)
                    blocks.append((s_index, cur_i, letter_char))
                    if s_index == 0:
                        break
                    next_c = from_prev_letter[cur_i][cur_c]
                    if next_c == -1 or next_c < 0 or next_c >= 26:
                        break
                    cur_i = s_index - 1
                    cur_c = next_c
                covered = [0] * n
                valid = True
                for s, e, _ in blocks:
                    if s < 0 or e >= n or s > e:
                        valid = False
                        break
                    for idx in range(s, e + 1):
                        if idx < 0 or idx >= n:
                            valid = False
                            break
                        covered[idx] = 1
                if not valid or sum(covered) != n:
                    continue
                blocks_sorted = sorted(blocks, key=lambda x: x[0])
                s_str = ''.join(letter * (e - s + 1) for s, e, letter in blocks_sorted)
                candidates.append(s_str)
        
        if not candidates:
            return ""
        return min(candidates)