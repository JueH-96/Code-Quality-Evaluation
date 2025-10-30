# YOUR CODE HERE
import sys
import heapq

def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    D = list(map(int, input[1:N+1]))
    L1, C1, K1 = map(int, input[N+1:N+4])
    L2, C2, K2 = map(int, input[N+4:N+7])

    # Create a list of (cost, length) pairs for each sensor type
    sensors = [(C1, L1), (C2, L2)]
    max_sensors = [K1, K2]

    # Initialize the dp array
    dp = [float('inf')] * (sum(D) + 1)
    dp[0] = 0

    # Fill the dp array
    for cost, length in sensors:
        for total_length in range(sum(D), 0, -1):
            for k in range(1, max_sensors[sensors.index((cost, length))] + 1):
                if total_length - k * length >= 0:
                    dp[total_length] = min(dp[total_length], dp[total_length - k * length] + k * cost)
                else:
                    break

    # Calculate the minimum cost to cover all sections
    min_cost = float('inf')
    for i in range(N):
        for total_length in range(D[i], sum(D) + 1):
            min_cost = min(min_cost, dp[total_length])

    # Check if it's possible to cover all sections
    if min_cost == float('inf'):
        print(-1)
    else:
        print(min_cost)

if __name__ == "__main__":
    main()