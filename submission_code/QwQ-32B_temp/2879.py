class Solution:
    def minimumChanges(self, s: str, k: int) -> int:
        n = len(s)
        max_L = n
        # Precompute divisors for each possible length L
        divisors_list = [[] for _ in range(max_L + 1)]
        for L in range(1, max_L + 1):
            divisors = set()
            for i in range(1, int(L**0.5) + 1):
                if L % i == 0:
                    divisors.add(i)
                    other = L // i
                    if other != L:
                        divisors.add(other)
            divisors.discard(L)
            divisors_list[L] = list(divisors)
        
        # Precompute cost array
        INF = float('inf')
        cost = [[INF for _ in range(n)] for _ in range(n)]
        for l in range(n):
            for r in range(l, n):
                L = r - l + 1
                if L < 2:
                    cost[l][r] = INF
                else:
                    min_cost = INF
                    divisors = divisors_list[L]
                    for d in divisors:
                        current_cost = 0
                        for m in range(d):
                            elements = []
                            current = l + m
                            while current <= r:
                                elements.append(s[current])
                                current += d
                            # Calculate cost for this group
                            n_group = len(elements)
                            cnt = 0
                            for i in range(n_group // 2):
                                if elements[i] != elements[n_group - 1 - i]:
                                    cnt += 1
                            current_cost += cnt
                        if current_cost < min_cost:
                            min_cost = current_cost
                    cost[l][r] = min_cost if min_cost != INF else INF
        
        # DP table initialization
        dp = [[INF] * (k + 1) for _ in range(n + 1)]
        dp[0][0] = 0
        for i in range(1, n + 1):
            for m in range(1, k + 1):
                if m == 1:
                    if i < 2:
                        dp[i][m] = INF
                    else:
                        dp[i][m] = cost[0][i-1]
                else:
                    start_j = 2 * (m - 1)
                    end_j = i - 2
                    if start_j > end_j:
                        dp[i][m] = INF
                    else:
                        min_val = INF
                        for j in range(start_j, end_j + 1):
                            if dp[j][m-1] + cost[j][i-1] < min_val:
                                min_val = dp[j][m-1] + cost[j][i-1]
                        dp[i][m] = min_val
        
        return dp[n][k]