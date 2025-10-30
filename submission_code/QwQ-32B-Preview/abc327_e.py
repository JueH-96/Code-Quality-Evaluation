import math

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    P = list(map(int, data[1:N+1]))
    
    # Precompute W(k) for k from 1 to N
    W = [10 * (1 - 0.9**k) for k in range(1, N+1)]
    
    # Initialize DP and max_prev
    DP = [[] for _ in range(N+1)]
    max_prev = [float('-inf')] * N
    
    # Iterate through each contest
    for j in range(1, N+1):
        for k in range(1, j+1):
            if k == 1:
                DP[j].append(P[j-1])
            else:
                DP[j].append(P[j-1] + 0.9 * max_prev[k-2])
            if k == 1:
                max_prev[k-1] = max(max_prev[k-1], DP[j][k-1])
            else:
                max_prev[k-1] = max(max_prev[k-1], DP[j][k-1])
    
    # Find the maximum R
    max_R = float('-inf')
    for k in range(1, N+1):
        # Find the maximum sum for subsets of size k
        max_sum = max(DP[j][k-1] for j in range(k, N+1))
        R = (max_sum / W[k-1]) - (1200 / math.sqrt(k))
        if R > max_R:
            max_R = R
    
    # Print the result with high precision
    print("{0:.12f}".format(max_R))

if __name__ == '__main__':
    main()