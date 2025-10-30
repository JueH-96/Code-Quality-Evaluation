n = int(input())
D = list(map(int, input().split()))
L1, C1, K1 = map(int, input().split())
L2, C2, K2 = map(int, input().split())

sections = []
possible = True

for D_i in D:
    candidates = set()
    
    # Source 1: 0 to min(5, K1)
    max_a1 = min(5, K1)
    for a in range(max_a1 + 1):
        candidates.add(a)
    
    # Source 2: around a_min1
    a_min1 = (D_i + L1 - 1) // L1 if L1 != 0 else 0
    for da in [-2, -1, 0, 1, 2]:
        a = a_min1 + da
        if a < 0:
            a = 0
        if a > K1:
            a = K1
        candidates.add(a)
    
    # Source 3: b from 0 to 5
    for b in range(6):
        required = D_i - b * L2
        if required <= 0:
            a_i = 0
        else:
            a_i = (required + L1 - 1) // L1
        a_i = max(0, min(a_i, K1))
        candidates.add(a_i)
    
    candidates = list(candidates)
    valid = []
    for a_i in candidates:
        if a_i * L1 + K2 * L2 >= D_i:
            if a_i * L1 >= D_i:
                b_i_min = 0
            else:
                required = D_i - a_i * L1
                b_i_min = (required + L2 - 1) // L2
            valid.append((a_i, b_i_min))
    
    if not valid:
        possible = False
    sections.append(valid)

if not possible:
    print(-1)
else:
    max_a = K1
    dp = [float('inf')] * (max_a + 1)
    dp[0] = 0
    
    for section in sections:
        new_dp = [float('inf')] * (max_a + 1)
        for a in range(max_a + 1):
            if dp[a] != float('inf'):
                for (a_i, b_i) in section:
                    new_a = a + a_i
                    if new_a > max_a:
                        continue
                    if new_dp[new_a] > dp[a] + b_i:
                        new_dp[new_a] = dp[a] + b_i
        dp = new_dp
    
    min_cost = float('inf')
    for a_total in range(max_a + 1):
        if dp[a_total] <= K2:
            cost = a_total * C1 + dp[a_total] * C2
            if cost < min_cost:
                min_cost = cost
    
    print(min_cost if min_cost != float('inf') else -1)