# YOUR CODE HERE
import sys

def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    courses = [(int(input[2*i+1]), int(input[2*i+2])) for i in range(N)]
    
    # Initialize dp arrays
    dp_healthy = [0] * (N + 1)
    dp_unhealthy = [float('-inf')] * (N + 1)
    
    for i in range(1, N + 1):
        x, y = courses[i-1]
        if x == 0:
            # Antidotal course
            dp_healthy[i] = max(dp_healthy[i-1] + y, dp_unhealthy[i-1] + y)
            dp_unhealthy[i] = dp_unhealthy[i-1]
        else:
            # Poisonous course
            dp_healthy[i] = dp_healthy[i-1]
            dp_unhealthy[i] = max(dp_healthy[i-1] + y, dp_unhealthy[i-1] + y)
    
    # The result is the maximum tastiness when Takahashi is healthy at the end
    result = max(dp_healthy[N], 0)
    print(result)

if __name__ == "__main__":
    main()