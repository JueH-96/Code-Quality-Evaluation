def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    D = list(map(int, input[idx:idx+N]))
    idx += N
    L1 = int(input[idx])
    C1 = int(input[idx+1])
    K1 = int(input[idx+2])
    idx += 3
    L2 = int(input[idx])
    C2 = int(input[idx+1])
    K2 = int(input[idx+2])
    idx += 3
    
    candidate_pairs = []
    
    for D_i in D:
        candidates = set()
        # Generate a from 0 to min(K1, 3)
        max_a_small = min(K1, 3)
        for a in range(0, max_a_small + 1):
            numerator = D_i - a * L1
            required_b = max(0, (numerator + L2 - 1) // L2) if numerator > 0 else 0
            if a * L1 + required_b * L2 >= D_i and required_b <= K2:
                candidates.add((a, required_b))
        # Generate b from 0 to min(K2, 3)
        max_b_small = min(K2, 3)
        for b in range(0, max_b_small + 1):
            numerator = D_i - b * L2
            required_a = max(0, (numerator + L1 - 1) // L1) if numerator > 0 else 0
            if required_a <= K1 and required_a * L1 + b * L2 >= D_i:
                candidates.add((required_a, b))
        # Generate a_max and around
        if L1 == 0:
            a_max = 0
        else:
            a_max = min(K1, (D_i + L1 - 1) // L1)
        for da in range(0, 4):
            a = a_max - da
            if a < 0:
                continue
            numerator = D_i - a * L1
            required_b = max(0, (numerator + L2 - 1) // L2) if numerator > 0 else 0
            if required_b <= K2 and a * L1 + required_b * L2 >= D_i:
                candidates.add((a, required_b))
        # Generate b_max and around
        if L2 == 0:
            b_max = 0
        else:
            b_max = min(K2, (D_i + L2 - 1) // L2)
        for db in range(0, 4):
            b = b_max - db
            if b < 0:
                continue
            numerator = D_i - b * L2
            required_a = max(0, (numerator + L1 - 1) // L1) if numerator > 0 else 0
            if required_a <= K1 and required_a * L1 + b * L2 >= D_i:
                candidates.add((required_a, b))
        # Generate small a and b combinations
        max_a = min(K1, 3)
        max_b = min(K2, 3)
        for a in range(0, max_a + 1):
            for b in range(0, max_b + 1):
                if a * L1 + b * L2 >= D_i:
                    candidates.add((a, b))
        # Convert to list and prune
        pairs = list(candidates)
        # Sort pairs by cost ascending, then a, then b
        pairs.sort(key=lambda x: (x[0] * C1 + x[1] * C2, x[0], x[1]))
        pruned = []
        for a, b in pairs:
            current_cost = a * C1 + b * C2
            dominated = False
            for pa, pb in pruned:
                pa_cost = pa * C1 + pb * C2
                if pa <= a and pb <= b and pa_cost <= current_cost:
                    dominated = True
                    break
            if not dominated:
                # Remove dominated pairs in pruned
                new_pruned = []
                for pa, pb in pruned:
                    pa_cost = pa * C1 + pb * C2
                    if not (a <= pa and b <= pb and current_cost <= pa_cost):
                        new_pruned.append((pa, pb))
                new_pruned.append((a, b))
                pruned = new_pruned
        candidate_pairs.append(pruned)
    
    # Initialize DP
    INF = float('inf')
    dp = [[INF] * (K2 + 1) for _ in range(K1 + 1)]
    dp[0][0] = 0
    
    for section in range(N):
        new_dp = [[INF] * (K2 + 1) for _ in range(K1 + 1)]
        pairs = candidate_pairs[section]
        for k1 in range(K1 + 1):
            for k2 in range(K2 + 1):
                if dp[k1][k2] == INF:
                    continue
                for (a, b) in pairs:
                    new_k1 = k1 + a
                    new_k2 = k2 + b
                    if new_k1 > K1 or new_k2 > K2:
                        continue
                    new_cost = dp[k1][k2] + a * C1 + b * C2
                    if new_cost < new_dp[new_k1][new_k2]:
                        new_dp[new_k1][new_k2] = new_cost
        dp = new_dp
    
    min_cost = INF
    for row in dp:
        current_min = min(row)
        if current_min < min_cost:
            min_cost = current_min
    print(-1 if min_cost == INF else min_cost)

if __name__ == "__main__":
    main()