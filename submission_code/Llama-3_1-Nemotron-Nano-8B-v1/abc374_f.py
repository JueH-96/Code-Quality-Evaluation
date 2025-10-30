n, k, x = map(int, input().split())
t = list(map(int, input().split()))
prefix = [0] * (n + 1)
for i in range(n):
    prefix[i+1] = prefix[i] + t[i]

INF = 10**18
dp = [[] for _ in range(n+1)]
dp[0].append((-INF, 0))

for i in range(n+1):
    for prev_ld, current_cost in dp[i]:
        for m in range(1, k+1):
            j = i + m
            if j > n:
                continue
            end = i + m - 1
            current_T_max = t[end]
            group_day = max(current_T_max, prev_ld + x)
            sum_T = prefix[end+1] - prefix[i]
            new_cost = current_cost + (group_day * m - sum_T)
            new_ld = group_day
            
            dominated = False
            to_remove = []
            for ld, cost in dp[j]:
                if ld <= new_ld and cost <= new_cost:
                    dominated = True
                    break
                if ld >= new_ld and cost >= new_cost:
                    to_remove.append((ld, cost))
            if dominated:
                continue
            
            for ld, cost in to_remove:
                if (ld, cost) in dp[j]:
                    dp[j].remove((ld, cost))
            
            dp[j].append((new_ld, new_cost))
            dp[j].sort()
            
            pruned = []
            for ld, cost in dp[j]:
                add = True
                for pld, pcost in pruned:
                    if pld <= ld and pcost <= cost:
                        add = False
                        break
                if add:
                    pruned.append((ld, cost))
            dp[j] = pruned

print(min(cost for _, cost in dp[n]))