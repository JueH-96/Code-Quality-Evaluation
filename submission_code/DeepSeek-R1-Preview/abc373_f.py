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
    
    dp = [-float('inf')] * (W + 1)
    dp[0] = 0  # Base case: 0 weight gives 0 happiness
    
    for w, v in items:
        for j in range(W, w - 1, -1):
            max_k = j // w
            if max_k == 0:
                k_options = [0, 1] if w <= j else [0]
            else:
                k_opt = v // 2
                k_opt = max(0, min(k_opt, max_k))
                k_options = []
                for delta in [-2, -1, 0, 1, 2]:
                    k_candidate = k_opt + delta
                    if 0 <= k_candidate <= max_k:
                        k_options.append(k_candidate)
                for delta in [-2, -1, 0, 1, 2]:
                    k_candidate = max_k + delta
                    if 0 <= k_candidate <= max_k:
                        k_options.append(k_candidate)
                k_options = list(set(k_options))  # Remove duplicates
                if 0 not in k_options:
                    k_options.append(0)
                if 1 not in k_options and max_k >= 1:
                    k_options.append(1)
                if 2 not in k_options and max_k >= 2:
                    k_options.append(2)
            current_max = dp[j]
            for k in k_options:
                if k * w > j:
                    continue
                prev = j - k * w
                if prev < 0:
                    continue
                if dp[prev] == -float('inf'):
                    continue
                candidate = dp[prev] + (k * v - k * k)
                if candidate > current_max:
                    current_max = candidate
            dp[j] = max(dp[j], current_max)
    
    max_happiness = max(dp)
    print(max_happiness)

if __name__ == '__main__':
    main()