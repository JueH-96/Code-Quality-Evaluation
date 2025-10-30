# YOUR CODE HERE
import sys
import threading
def main():
    import sys
    import math
    sys.setrecursionlimit(1 << 25)
    N_and_rest = sys.stdin.read().split()
    N = int(N_and_rest[0])
    A = list(map(int, N_and_rest[1:N+1]))
    mod = 998244353
    N = len(A)
    MAX_CNT = 1000  # Adjust if necessary based on practical observations
    from collections import defaultdict

    DP = [{} for _ in range(N+1)]
    DP[0][1] = 1  # cnt_0 = 1
    for i in range(N):
        dp_curr = DP[i]
        dp_next = {}
        ai = A[i]
        for cnt, ways in dp_curr.items():
            for eqflag in [True, False]:
                # For current cnt and eqflag, process possible V_i
                minvi = max(1 - cnt, 0)
                maxvi = ai if eqflag else min(ai, MAX_CNT)
                # Limit maxvi to prevent excessive computation
                # Since V_i can be large, we limit maxvi
                # Alternatively, process only up to a reasonable limit
                max_vi_to_consider = min(maxvi, minvi + 20)  # Limit the range to minvi + 20
                for vi in range(minvi, max_vi_to_consider +1):
                    cnt_next = cnt -1 + vi
                    if cnt_next < 0 or cnt_next > MAX_CNT:
                        continue
                    eqflag_next = eqflag and (vi == ai)
                    key = cnt_next
                    state = (eqflag_next)
                    dp_state = (key, state)
                    if dp_state not in dp_next:
                        dp_next[dp_state] = 0
                    dp_next[dp_state] = (dp_next[dp_state] + ways) % mod
                # Handle vi == maxvi separately if maxvi > max_vi_to_consider
                if maxvi > max_vi_to_consider:
                    vi = maxvi
                    cnt_next = cnt -1 + vi
                    if cnt_next >= 0 and cnt_next <= MAX_CNT:
                        eqflag_next = eqflag and (vi == ai)
                        key = cnt_next
                        state = (eqflag_next)
                        dp_state = (key, state)
                        if dp_state not in dp_next:
                            dp_next[dp_state] = 0
                        dp_next[dp_state] = (dp_next[dp_state] + ways) % mod
        DP[i+1] = dp_next
    # At the end, cnt_N must be 0
    ans = 0
    for (cnt, eqflag), ways in DP[N].items():
        if cnt == 0:
            ans = (ans + ways) % mod
    print(ans)
if __name__ == '__main__':
    threading.Thread(target=main).start()