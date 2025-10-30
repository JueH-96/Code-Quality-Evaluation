# YOUR CODE HERE
import sys
import math

import threading

def main():
    import sys
    import math
    sys.setrecursionlimit(1 << 25)
    N_and_rest = sys.stdin.read().split()
    N = int(N_and_rest[0])
    P = list(map(int, N_and_rest[1:N+1]))

    N = len(P)
    N_orig = N

    dp_s = [0.0] * (N + 1)  # dp_s[k]: maximum s achievable by selecting k contests
    dp_d = [0.0] * (N + 1)  # dp_d[k]: corresponding d for dp_s[k]
    dp_R = [-math.inf] * (N + 1)  # dp_R[k]: maximum R achievable by selecting k contests

    dp_s[0] = 0.0
    dp_d[0] = 0.0
    dp_R[0] = -math.inf

    max_R = -math.inf

    for i in range(N):
        P_i = P[i]
        # We need to process k from i+1 down to 1
        # To prevent overwriting dp_s[k -1] before it's used
        # We copy dp_s and dp_d to temporary arrays
        dp_s_prev = dp_s[:]
        dp_d_prev = dp_d[:]
        for k in range(min(i + 1, N), 0, -1):
            s_candidate = dp_s_prev[k - 1] * 0.9 + P_i
            d_candidate = dp_d_prev[k - 1] * 0.9 + 1.0
            R_candidate = s_candidate / d_candidate - 1200.0 / math.sqrt(k)
            if R_candidate > dp_R[k]:
                dp_s[k] = s_candidate
                dp_d[k] = d_candidate
                dp_R[k] = R_candidate
                if R_candidate > max_R:
                    max_R = R_candidate
            elif R_candidate == dp_R[k]:
                if s_candidate > dp_s[k]:  # If same R, keep the one with larger s
                    dp_s[k] = s_candidate
                    dp_d[k] = d_candidate
        # We can ignore the case where we don't include P_i, as dp_s[k] and dp_d[k] remain the same
    # After processing all positions, the maximum R is in dp_R[k], where k ranges from 1 to N
    print("%.15f" % max_R)
    
def threading_main():
    threading.Thread(target=main).start()

threading_main()