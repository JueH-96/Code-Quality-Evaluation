import sys

def main():
    input = sys.stdin.read
    data = input().split()

    N = int(data[0])
    K = int(data[1])
    X = [int(data[i]) for i in range(2, 2 + N)]
    A = [int(data[i]) for i in range(2 + N, 2 + 2 * N)]

    # Convert 1-based index to 0-based index
    X = [x - 1 for x in X]

    # Simulate the process using cycle detection
    seen = {}
    current = A
    index = 0

    while index < K:
        if tuple(current) in seen:
            cycle_length = index - seen[tuple(current)]
            K_remaining = (K - index) % cycle_length
            index = K - K_remaining
        else:
            seen[tuple(current)] = index
            current = [current[X[i]] for i in range(N)]
            index += 1

    result = current
    print(' '.join(map(str, result)))

if __name__ == "__main__":
    main()