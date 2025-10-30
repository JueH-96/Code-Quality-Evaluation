import sys

def main():
    import sys
    import math
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    M = int(input[ptr])
    ptr += 1

    wheels = []
    for _ in range(N):
        C_i = int(input[ptr])
        ptr += 1
        P_i = int(input[ptr])
        ptr += 1
        S_list = list(map(int, input[ptr:ptr+P_i]))
        ptr += P_i
        wheels.append((C_i, P_i, S_list))
    
    dp = [0.0] * (M + 1)
    for x in range(M-1, -1, -1):
        best = float('inf')
        for (C, P, S_list) in wheels:
            sum_rest = 0.0
            k = 0
            for s in S_list:
                if s == 0:
                    k += 1
                else:
                    next_state = x + s
                    if next_state >= M:
                        continue
                    sum_rest += dp[next_state]
            denominator = 1.0 - (k / P)
            numerator = C + (sum_rest / P)
            val = numerator / denominator
            if val < best:
                best = val
        dp[x] = best
    
    print("{0:.20f}".format(dp[0]))

if __name__ == "__main__":
    main()