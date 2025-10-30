import math

def main():
    import sys
    input = sys.stdin.read().split()
    n = int(input[0])
    P = list(map(int, input[1:n+1]))
    
    if n == 0:
        print(0.0)
        return
    
    max_r = -float('inf')
    dp_prev = [0.0] * n
    den_prev = [1.0] * n
    
    for j in range(1, n+1):
        if j == 1:
            for i in range(n):
                r = P[i] - 1200.0 / math.sqrt(1)
                if r > max_r:
                    max_r = r
        else:
            current_dp = [0.0] * n
            current_den = [0.0] * n
            current_max_dp = 0.0
            current_max_den = 0.0
            
            for i in range(j-1, n):
                current_max_dp = max(current_max_dp, dp_prev[i-1] * 0.9)
                current_max_den = max(current_max_den, den_prev[i-1] * 0.9)
                
                current_dp[i] = current_max_dp + P[i]
                current_den[i] = current_max_den + 1.0
                
                dp_prev[i] = current_dp[i]
                den_prev[i] = current_den[i]
            
            for i in range(j-1, n):
                numerator = current_dp[i]
                denominator = current_den[i]
                r = (numerator / denominator) - (1200.0 / math.sqrt(j))
                if r > max_r:
                    max_r = r
    
    print("{0:.15f}".format(max_r))

if __name__ == '__main__':
    main()