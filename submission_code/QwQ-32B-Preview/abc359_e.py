def main():
    import sys
    N_and_rest = sys.stdin.read().split()
    N = int(N_and_rest[0])
    H = list(map(int, N_and_rest[1:N+1]))
    
    ans = [0] * N
    max_h = 0
    for i in range(N):
        max_h = max(max_h, H[i])
        ans[i] = max_h + i + 1
        print(ans[i], end=' ' if i < N-1 else '
')

if __name__ == "__main__":
    main()