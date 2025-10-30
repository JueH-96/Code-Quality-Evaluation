def main():
    import sys
    input = sys.stdin.readline

    N, M = map(int, input().split())
    S = [input().rstrip() for _ in range(N)]
    T = [input().rstrip() for _ in range(M)]

    # Try every possible top-left (a, b) of an MxM subgrid in S
    for a in range(N - M + 1):
        for b in range(N - M + 1):
            match = True
            for i in range(M):
                # Compare the substring of length M starting at column b
                if S[a + i][b:b + M] != T[i]:
                    match = False
                    break
            if match:
                # Convert to 1-based indexing and print
                print(a + 1, b + 1)
                return

if __name__ == "__main__":
    main()