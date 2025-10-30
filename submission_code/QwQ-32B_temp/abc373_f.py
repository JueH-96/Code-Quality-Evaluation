import sys

def main():
    N, W = map(int, sys.stdin.readline().split())
    items = []
    for _ in range(N):
        w, v = map(int, sys.stdin.readline().split())
        items.append((w, v))
    
    INF = float('-inf')
    dp = [INF] * (W + 1)
    dp[0] = 0
    
    for w, v in items:
        K_i = min(v // 2, W // w)
        # Iterate backwards to prevent overwriting
        for j in range(W, -1, -1):
            if dp[j] != INF:
                max_k = min(K_i, (W - j) // w)
                if max_k > 0:
                    new_weight = j + max_k * w
                    new_val = dp[j] + max_k * (v - max_k)
                    if new_val > dp[new_weight]:
                        dp[new_weight] = new_val
    
    print(max(dp))

if __name__ == "__main__":
    main()