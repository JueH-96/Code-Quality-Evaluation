# YOUR CODE HERE
import sys

def min_cost(N, S, M, L):
    min_cost = float('inf')
    for x in range(N // 12 + 1):
        for y in range((N - 12 * x) // 8 + 1):
            z = (N - 12 * x - 8 * y + 5) // 6  # Ceiling division
            cost = x * L + y * M + z * S
            if cost < min_cost:
                min_cost = cost
    return min_cost

def main():
    input = sys.stdin.read().strip().split()
    N = int(input[0])
    S = int(input[1])
    M = int(input[2])
    L = int(input[3])
    print(min_cost(N, S, M, L))

if __name__ == "__main__":
    main()