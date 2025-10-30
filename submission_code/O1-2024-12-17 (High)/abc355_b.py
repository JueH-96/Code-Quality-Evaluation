def main():
    import sys
    data = sys.stdin.read().strip().split()
    N, M = map(int, data[:2])
    A = list(map(int, data[2:2+N]))
    B = list(map(int, data[2+N:2+N+M]))

    combined = A + B
    combined.sort()

    A_set = set(A)
    for i in range(N + M - 1):
        if combined[i] in A_set and combined[i+1] in A_set:
            print("Yes")
            return

    print("No")

# Do not forget to call main()
if __name__ == "__main__":
    main()