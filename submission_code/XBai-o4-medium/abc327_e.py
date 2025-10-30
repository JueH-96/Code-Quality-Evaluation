import math
import sys

def main():
    input = sys.stdin.read().split()
    n = int(input[0])
    p = list(map(int, input[1:n+1]))
    
    pow_09 = [1.0] * (n + 1)
    for i in range(1, n+1):
        pow_09[i] = pow_09[i-1] * 0.9
    
    dp = [ -float('inf') ] * (n + 1)
    dp[0] = 0.0
    
    for i in range(n):
        current_p = p[i]
        # j ranges from i+1 down to 1
        for j in range(i+1, 0, -1):
            if j <= n:
                if dp[j] < dp[j-1] * 0.9 + current_p:
                    dp[j] = dp[j-1] * 0.9 + current_p
    
    max_r = -float('inf')
    for j in range(1, n+1):
        denominator = 10.0 * (1.0 - pow_09[j])
        w = dp[j] / denominator
        penalty = 1200.0 / math.sqrt(j)
        current_r = w - penalty
        if current_r > max_r:
            max_r = current_r
    
    print("{0:.15f}".format(max_r))
    
if __name__ == '__main__':
    main()