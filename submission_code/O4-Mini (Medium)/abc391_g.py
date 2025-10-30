import sys
import threading
def main():
    import sys
    sys.setrecursionlimit(1<<25)
    mod = 998244353

    data = sys.stdin.read().strip().split()
    N = int(data[0]); M = int(data[1])
    S = data[2].strip()
    # Precompute transitions for LCS DP rows encoded as bitmask of length N.
    # mask bit i (0-based) means dp[i+1] = dp[i] + 1, else same as dp[i].
    # We'll build for each mask and each character c the new mask.
    # Then group by target mask counts.

    # Precompute dp_prefix values for each mask and each prefix length
    # Actually we'll compute on the fly dp_prev from mask.
    # Build transition_counts[mask] = dict {new_mask: count_of_letters}
    transition_counts = [None] * (1<<N)
    # For speed, prepare list of ordinals of S
    S_chars = [ord(c) for c in S]

    for mask in range(1<<N):
        # Reconstruct dp_prev array of length N+1
        # dp_prev[0] = 0; dp_prev[i] = sum of bits 0..i-1 of mask
        dp_prev = [0] * (N+1)
        cnt = 0
        for i in range(1, N+1):
            # check bit i-1
            if (mask >> (i-1)) & 1:
                cnt += 1
            dp_prev[i] = cnt
        cnts = {}
        # For each letter
        for oc in range(26):
            # character ordinal for comparison
            c_ord = ord('a') + oc
            # build dp_new
            dp_new0 = 0
            new_mask = 0
            # dp_new[0] = 0
            for i in range(1, N+1):
                # take max from top or left
                # top = dp_prev[i], left = dp_new0
                v = dp_prev[i] if dp_prev[i] >= dp_new0 else dp_new0
                # if match, consider dp_prev[i-1]+1
                if S_chars[i-1] == c_ord:
                    tmp = dp_prev[i-1] + 1
                    if tmp > v:
                        v = tmp
                # dp_new[i] = v
                # if dp_new[i] = dp_new[i-1]+1 then bit i-1 = 1
                if v == dp_new0 + 1:
                    # increment, set bit
                    new_mask |= (1 << (i-1))
                    dp_new0 = dp_new0 + 1
                else:
                    # stays same as dp_new0
                    dp_new0 = dp_new0 if dp_new0 >= dp_prev[i] else dp_prev[i]
                    # but dp_new0 should equal v; simpler set directly
                    dp_new0 = v
                # proceed
            # record transition
            cnts[new_mask] = cnts.get(new_mask, 0) + 1
        transition_counts[mask] = list(cnts.items())

    # DP over T of length M
    size = 1<<N
    dp_cur = [0] * size
    dp_cur[0] = 1
    for _ in range(M):
        dp_next = [0] * size
        for mask, val in enumerate(dp_cur):
            if val:
                for (nmask, cnt) in transition_counts[mask]:
                    dp_next[nmask] = (dp_next[nmask] + val * cnt) % mod
        dp_cur = dp_next

    # Collect answers
    ans = [0] * (N+1)
    for mask, val in enumerate(dp_cur):
        if val:
            k = mask.bit_count()
            ans[k] = (ans[k] + val) % mod

    # Output ans[0..N]
    print(" ".join(str(ans[k]) for k in range(N+1)))

if __name__ == "__main__":
    main()