# YOUR CODE HERE
import sys

def main():
    import sys
    import math

    input = sys.stdin.read
    data = input().split()
    
    N, K = int(data[0]), int(data[1])
    A_B = list(map(int, data[2:]))
    functions = [(A_B[i], A_B[i+1]) for i in range(0, 2*N, 2)]
    
    dp = [-math.inf] * (K +1)
    dp[0] = 1
    
    for A,B in functions:
        for k in range(min(K, 10), 0, -1):
            if dp[k-1] != -math.inf:
                candidate = A * dp[k-1] + B
                if candidate > dp[k]:
                    dp[k] = candidate
    print(int(dp[K]))

if __name__ == "__main__":
    main()