def main():
    import sys
    input = sys.stdin.read().split()
    n = int(input[0])
    H = list(map(int, input[1:n+1]))
    
    ans = []
    prev_prev = 0  # ans[i-2]
    prev = 0       # ans[i-1]
    if n >= 1:
        # ans[0] is not used, but for i=1 (1-based), we need to compute ans[1]
        # which is H[0] + 1
        ans.append(H[0] + 1)
        prev = ans[-1]
        prev_prev = 0
    for i in range(1, n):
        current = prev + max(H[i] - (prev - prev_prev), 0) + 1
        ans.append(current)
        prev_prev, prev = prev, current
    print(' '.join(map(str, ans)))

if __name__ == "__main__":
    main()