def main():
    import sys
    input = sys.stdin.readline

    # Read input
    N, M = map(int, input().split())
    A = list(map(int, input().split()))

    # Mark firework days
    firework = [False] * (N + 1)
    for a in A:
        firework[a] = True

    # ans[i] will hold days until next firework from day i
    ans = [0] * (N + 1)

    # The last day always has a firework, so ans[N] = 0
    # Fill backwards
    for i in range(N - 1, 0, -1):
        if firework[i]:
            ans[i] = 0
        else:
            ans[i] = ans[i + 1] + 1

    # Output results
    out = sys.stdout.write
    for i in range(1, N + 1):
        out(str(ans[i]) + "
")

if __name__ == "__main__":
    main()