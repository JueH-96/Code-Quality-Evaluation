def main():
    import sys
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    T = int(data[1])
    A = list(map(int, data[2:]))

    row_count = [0] * N
    col_count = [0] * N
    diag_main = 0
    diag_anti = 0

    for i in range(T):
        a = A[i]
        r = (a - 1) // N
        c = (a - 1) % N

        row_count[r] += 1
        col_count[c] += 1
        
        if r == c:
            diag_main += 1
        if r + c == N - 1:
            diag_anti += 1

        if (
            row_count[r] == N or
            col_count[c] == N or
            diag_main == N or
            diag_anti == N
        ):
            print(i + 1)
            return

    print(-1)

# Do not forget to call main().
if __name__ == "__main__":
    main()