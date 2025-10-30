def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    W = int(input[idx])
    idx += 1
    items = []
    for _ in range(N):
        w = int(input[idx])
        idx += 1
        v = int(input[idx])
        idx += 1
        items.append((w, v))
    
    dp = [0] * (W + 1)
    
    for w, v in items:
        for j in range(W, -1, -1):
            x_candidate = min(v // 2, j // w)
            x_min = max(0, x_candidate - 2)
            x_max = min(x_candidate + 2, j // w)
            max_val = dp[j]
            for x in range(x_min, x_max + 1):
                prev = j - x * w
                if prev >= 0:
                    current = dp[prev] + x * v - x * x
                    if current > max_val:
                        max_val = current
            dp[j] = max_val
    
    print(max(dp))

if __name__ == "__main__":
    main()