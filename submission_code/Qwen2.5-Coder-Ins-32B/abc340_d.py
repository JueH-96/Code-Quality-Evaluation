# YOUR CODE HERE
import sys
import heapq

def main():
    import sys
    import heapq
    input = sys.stdin.read().split()
    N = int(input[0])
    A = []
    B = []
    X = []
    index = 1
    for i in range(N-1):
        A.append(int(input[index]))
        B.append(int(input[index+1]))
        X.append(int(input[index+2]))
        index += 3

    # Initialize the dp array with infinity
    dp = [float('inf')] * (N + 1)
    dp[1] = 0
    # Min-heap to store (time, stage)
    heap = [(0, 1)]

    while heap:
        current_time, current_stage = heapq.heappop(heap)
        if current_time > dp[current_stage]:
            continue
        # Option 1: Spend A_i seconds to clear stage i, which allows you to play stage i+1.
        if current_stage < N:
            if current_time + A[current_stage - 1] < dp[current_stage + 1]:
                dp[current_stage + 1] = current_time + A[current_stage - 1]
                heapq.heappush(heap, (dp[current_stage + 1], current_stage + 1))
        # Option 2: Spend B_i seconds to clear stage i, which allows you to play stage X_i.
        if X[current_stage - 1] <= N:
            if current_time + B[current_stage - 1] < dp[X[current_stage - 1]]:
                dp[X[current_stage - 1]] = current_time + B[current_stage - 1]
                heapq.heappush(heap, (dp[X[current_stage - 1]], X[current_stage - 1]))

    print(dp[N])

if __name__ == "__main__":
    main()