import sys

def main():
    input = sys.stdin.read
    data = input().split()

    N = int(data[0])
    K = int(data[1])
    P = list(map(int, data[2:]))

    # Convert P to zero-based index for easier handling
    P = [x - 1 for x in P]

    # Create a list to store the next position for each element
    next_position = [0] * N
    for i in range(N):
        next_position[P[i]] = i

    # Simulate the operations
    current_position = list(range(N))
    for _ in range(K):
        new_position = [next_position[current_position[i]] for i in range(N)]
        current_position = new_position

    # Convert back to one-based index for the final output
    result = [current_position[i] + 1 for i in range(N)]

    # Print the result
    print(" ".join(map(str, result)))

if __name__ == "__main__":
    main()