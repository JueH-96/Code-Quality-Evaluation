import math

def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    P = list(map(int, input[1:N+1]))
    
    dp = [-float('inf')] * (N + 1)
    dp[0] = 0.0
    
    for p in P:
        current_max_k = 0
        for k in range(len(dp)-1, 0, -1):
            if dp[k] != -float('inf'):
                current_max_k = k
                break
        max_k = min(current_max_k + 1, N)
        for k in range(max_k, 0, -1):
            if k == 1:
                candidate = p
            else:
                if dp[k-1] == -float('inf'):
                    continue
                candidate = 0.9 * dp[k-1] + p
            if candidate > dp[k]:
                dp[k] = candidate
    
    max_R = -float('inf')
    for k in range(1, N+1):
        if dp[k] == -float('inf'):
            continue
        denominator = 10.0 * (1.0 - (0.9 ** k))
        avg = dp[k] / denominator
        penalty = 1200.0 / math.sqrt(k)
        R = avg - penalty
        if R > max_R:
            max_R = R
    
    print("{0:.12f}".format(max_R))

if __name__ == '__main__':
    main()