class Solution:
    def countCompleteSubstrings(self, word: str, k: int) -> int:
        n = len(word)
        # Precompute bad adjacent pairs: bad_ps[i] = number of bad pairs up to index i-1
        bad_ps = [0] * n
        for i in range(1, n):
            if abs(ord(word[i]) - ord(word[i-1])) > 2:
                bad_ps[i] = bad_ps[i-1] + 1
            else:
                bad_ps[i] = bad_ps[i-1]
        
        ans = 0
        max_m = min(26, n // k)
        # For each possible number of distinct chars m
        for m in range(1, max_m + 1):
            L = m * k
            if L > n:
                break
            counts = [0] * 26
            cnt_eq_k = 0
            cnt_over_k = 0
            
            # initialize first window [0..L-1]
            for j in range(L):
                idx = ord(word[j]) - ord('a')
                c_before = counts[idx]
                counts[idx] += 1
                c_after = counts[idx]
                # track transitions around k
                if c_before == k - 1 and c_after == k:
                    cnt_eq_k += 1
                elif c_before == k and c_after == k + 1:
                    cnt_eq_k -= 1
                    cnt_over_k += 1
                elif c_before >= k + 1:
                    # still over k
                    pass
            
            # check windows
            for i in range(0, n - L + 1):
                # window is [i..i+L-1]
                # check no count > k, exactly m chars with count==k, and no bad adjacents
                if cnt_over_k == 0 and cnt_eq_k == m:
                    # bad pairs in [i..i+L-1] span indices i..i+L-2
                    if bad_ps[i+L-1] - bad_ps[i] == 0:
                        ans += 1
                # slide: remove word[i], add word[i+L]
                if i + L < n:
                    # remove
                    idx_r = ord(word[i]) - ord('a')
                    c_before = counts[idx_r]
                    counts[idx_r] -= 1
                    c_after = counts[idx_r]
                    if c_before == k and c_after == k - 1:
                        cnt_eq_k -= 1
                    elif c_before == k + 1 and c_after == k:
                        cnt_eq_k += 1
                        cnt_over_k -= 1
                    # elif c_before > k + 1: still over k, no change
                    # add
                    idx_a = ord(word[i+L]) - ord('a')
                    c_before = counts[idx_a]
                    counts[idx_a] += 1
                    c_after = counts[idx_a]
                    if c_before == k - 1 and c_after == k:
                        cnt_eq_k += 1
                    elif c_before == k and c_after == k + 1:
                        cnt_eq_k -= 1
                        cnt_over_k += 1
                    # elif c_before >= k + 1: still over k, no change
            
        return ans