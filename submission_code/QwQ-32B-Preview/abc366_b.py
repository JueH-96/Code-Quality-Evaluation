def main():
    import sys

    # Read input
    input = sys.stdin.read
    data = input().splitlines()
    N = int(data[0])
    strings = data[1:N+1]

    # Find the maximum length M
    M = max(len(s) for s in strings)

    # Initialize list to hold Tj strings
    T = [''] * M

    # Build each Tj
    for j in range(M):
        for i in range(N-1, -1, -1):  # From N-1 down to 0
            if j < len(strings[i]):
                T[j] += strings[i][j]
            else:
                T[j] += '*'
        # Remove trailing '*' from Tj
        T[j] = T[j].rstrip('*')

    # Print each Tj
    for t in T:
        print(t)

if __name__ == "__main__":
    main()