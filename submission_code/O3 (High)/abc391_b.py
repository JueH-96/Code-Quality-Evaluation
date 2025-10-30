import sys

def main() -> None:
    # Read input
    data = sys.stdin.read().strip().splitlines()
    if not data:
        return
    N, M = map(int, data[0].split())
    S = data[1:1 + N]
    T = data[1 + N:1 + N + M]

    # Iterate over every possible top-left corner (a-1, b-1)
    for a0 in range(N - M + 1):          # 0-based row index
        for b0 in range(N - M + 1):      # 0-based column index
            ok = True
            for i in range(M):
                if S[a0 + i][b0:b0 + M] != T[i]:
                    ok = False
                    break
            if ok:
                # Convert to 1-based indices
                print(a0 + 1, b0 + 1)
                return

if __name__ == "__main__":
    main()