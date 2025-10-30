def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    W = int(input[idx])
    idx += 1
    types = []
    for _ in range(N):
        w = int(input[idx])
        v = int(input[idx+1])
        types.append((w, v))
        idx += 2
    
    INF = -1 << 60
    dp = [INF] * (W + 1)
    dp[0] = 0
    
    for w_i, v_i in types:
        tmp_dp = dp.copy()
        for w in range(W + 1):
            if dp[w] == INF:
                continue
            remaining = W - w
            if remaining < 0:
                max_k = 0
            else:
                max_k = remaining // w_i
            max_k = max(max_k, 0)
            candidates = [max_k, v_i // 2, (v_i + 1) // 2]
            clamped = set()
            for k in candidates:
                clamped_k = min(k, max_k)
                clamped_k = max(clamped_k, 0)
                clamped.add(clamped_k)
            for ck in clamped:
                new_w = w + ck * w_i
                if new_w > W:
                    continue
                new_val = dp[w] + ck * v_i - ck * ck
                if new_val > tmp_dp[new_w]:
                    tmp_dp[new_w] = new_val
        dp = tmp_dp
    
    print(max(dp))

if __name__ == "__main__":
    main()