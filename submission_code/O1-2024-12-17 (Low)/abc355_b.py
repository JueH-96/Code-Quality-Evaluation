def main():
    import sys

    data = sys.stdin.read().strip().split()
    N, M = map(int, data[:2])
    A = list(map(int, data[2:2+N]))
    B = list(map(int, data[2+N:2+N+M]))

    # Convert A and B to sets to quickly check membership
    A_set = set(A)

    # Merge and sort
    C = sorted(A + B)

    # Check consecutive elements
    for i in range(N+M - 1):
        if C[i] in A_set and C[i+1] in A_set:
            print("Yes")
            return

    print("No")

# Do not forget to call main!
if __name__ == "__main__":
    main()