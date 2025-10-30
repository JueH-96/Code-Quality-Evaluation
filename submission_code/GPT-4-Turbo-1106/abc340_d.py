import sys

def main():
    N = int(input().strip())
    stages = [list(map(int, input().split())) for _ in range(N - 1)]

    # Initialize the time required to reach each stage
    time_to_reach = [float('inf')] * N
    time_to_reach[0] = 0  # It takes 0 seconds to be at stage 1

    for i in range(N - 1):
        A, B, X = stages[i]
        # Update the time to reach the next stage by clearing this stage with A_i seconds
        time_to_reach[i + 1] = min(time_to_reach[i + 1], time_to_reach[i] + A)
        # Update the time to reach the X_i stage by clearing this stage with B_i seconds
        if X - 1 < N:
            time_to_reach[X - 1] = min(time_to_reach[X - 1], time_to_reach[i] + B)

    # The time to reach the last stage is the answer
    print(time_to_reach[-1])

if __name__ == "__main__":
    main()