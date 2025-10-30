class Solution:
    def makeStringGood(self, s: str) -> int:
        # Count occurrences of each letter
        cnt = [0] * 26
        for ch in s:
            cnt[ord(ch) - ord('a')] += 1
        N = len(s)
        
        # Build candidate k values: include each distinct count and count+1, plus 1
        uniq = set(cnt)
        candidate_k = set()
        candidate_k.add(1)
        for v in uniq:
            if v > 0:
                candidate_k.add(v)
            candidate_k.add(v + 1)
        # filter k>=1
        candidate_k = [k for k in candidate_k if k >= 1]
        
        INF = 10**18
        ans = INF
        
        # For each possible number of distinct letters m = 1..26
        for m in range(1, 27):
            # For each candidate uniform frequency k
            for k in candidate_k:
                # We require exactly m*k final length (some insertions/deletions)
                # Precompute arrays used in DP:
                # base_gain[c] = 2 * min(cnt[c], k)
                # surplus_if_sel[c] = max(cnt[c] - k, 0)
                # deficit[c] = max(k - cnt[c], 0)
                base_gain = [0] * 26
                surplus_if_sel = [0] * 26
                deficit = [0] * 26
                for i in range(26):
                    ci = cnt[i]
                    if ci < k:
                        base_gain[i] = 2 * ci
                        deficit[i] = k - ci
                        surplus_if_sel[i] = 0
                    else:
                        base_gain[i] = 2 * k
                        deficit[i] = 0
                        surplus_if_sel[i] = ci - k
                
                # DP over the 26 letters choosing exactly m of them
                # dp_prev[s][prev_sel] = best score (2*matches + neighbor savings)
                # after processing up to letter idx-1, having picked s so far,
                # and prev_sel = 0/1 tells if letter idx-1 was selected.
                dp_prev = [[-INF, -INF] for _ in range(m + 1)]
                dp_prev[0][0] = 0
                
                # iterate letters 0..25
                for c in range(26):
                    dp_curr = [[-INF, -INF] for _ in range(m + 1)]
                    # letter index c; we decide to select it or not
                    for s_count in range(m + 1):
                        for prev_sel in (0, 1):
                            val = dp_prev[s_count][prev_sel]
                            if val < 0:
                                continue
                            # Case 1: do not select letter c
                            # selection stays s_count, and prev_sel for next is 0
                            if val > dp_curr[s_count][0]:
                                dp_curr[s_count][0] = val
                            # Case 2: select letter c (if we still need picks)
                            if s_count < m:
                                # gain from matching same-letter
                                gain = base_gain[c]
                                # neighbor saving from c-1 -> c
                                if c > 0:
                                    # surplus at c-1 depending on whether it was selected
                                    if prev_sel == 1:
                                        prev_surp = surplus_if_sel[c-1]
                                    else:
                                        prev_surp = cnt[c-1]
                                    # deficit at c
                                    cur_def = deficit[c]
                                    # saving is min of those
                                    if prev_surp > 0 and cur_def > 0:
                                        gain += min(prev_surp, cur_def)
                                # update dp
                                new_val = val + gain
                                if new_val > dp_curr[s_count+1][1]:
                                    dp_curr[s_count+1][1] = new_val
                    dp_prev = dp_curr
                
                # after c=25, look at dp_prev[m][*]
                best_F = dp_prev[m][0]
                if dp_prev[m][1] > best_F:
                    best_F = dp_prev[m][1]
                if best_F < 0:
                    # no valid selection of size m
                    continue
                # total cost = N + (m*k) - best_F
                cost = N + m * k - best_F
                if cost < ans:
                    ans = cost
        
        return ans