def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    X = int(input[idx])
    idx += 1
    Y = int(input[idx])
    idx += 1
    dishes = []
    for _ in range(N):
        a = int(input[idx])
        idx += 1
        b = int(input[idx])
        idx += 1
        dishes.append((a, b))
    
    # Initialize DP: dp[k] is a dictionary mapping sum_b to min_sum_a
    dp = [{} for _ in range(N+1)]
    dp[0][0] = 0
    
    for a, b in dishes:
        # Process in reverse order to avoid reusing the same dish multiple times
        for k in range(N-1, -1, -1):
            if not dp[k]:
                continue
            for sum_b in list(dp[k].keys()):
                current_sum_a = dp[k][sum_b]
                new_sum_a = current_sum_a + a
                new_sum_b = sum_b + b
                new_k = k + 1
                if new_k > N:
                    continue
                if new_sum_b > Y or new_sum_a > X:
                    continue
                if new_sum_b in dp[new_k]:
                    if new_sum_a < dp[new_k][new_sum_b]:
                        dp[new_k][new_sum_b] = new_sum_a
                else:
                    dp[new_k][new_sum_b] = new_sum_a
    
    # Find the maximum k0 where there exists a valid sum_a and sum_b
    k0 = -1
    for k in range(N, -1, -1):
        if dp[k]:
            k0 = k
            break
    
    # The answer is min(k0 + 1, N)
    print(min(k0 + 1, N))

if __name__ == '__main__':
    main()