import sys
input = sys.stdin.read

def find_cycle(N, A):
    # Initialize variables
    index = 0
    visited = [False] * (N + 1)
    result = []

    # Start from any vertex
    while True:
        visited[index] = True
        result.append(index)
        index = A[index]

        # Check if we have visited this vertex before
        if visited[index]:
            # Find the start of the cycle
            start = result.index(index)
            cycle = result[start:]
            return cycle

def main():
    data = input().split()
    N = int(data[0])
    A = [0] + [int(data[i]) for i in range(1, N + 1)]  # 1-indexed

    cycle = find_cycle(N, A)

    # Output the result
    print(len(cycle))
    print(' '.join(map(str, cycle)))

if __name__ == "__main__":
    main()