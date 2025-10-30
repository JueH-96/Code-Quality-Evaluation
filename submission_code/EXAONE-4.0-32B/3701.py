class Solution:
    def minCostGoodCaption(self, caption: str) -> str:
        n = len(caption)
        if n < 3:
            return ""
        
        INF = 10**18
        dp = [[[INF] * 3 for _ in range(26)] for __ in range(n)]
        parent = [[[None] * 3 for _ in range(26)] for __ in range(n)]
        
        for c in range(26):
            cost0 = abs(ord(caption[0]) - (ord('a') + c))
            dp[0][c][0] = cost0
        
        for i in range(1, n):
            for c in range(26):
                current_char_cost = abs(ord(caption[i]) - (ord('a') + c))
                for prev_char in range(26):
                    for prev_state in range(3):
                        if dp[i-1][prev_char][prev_state] == INF:
                            continue
                        total_cost = dp[i-1][prev_char][prev_state] + current_char_cost
                        if prev_char == c:
                            if prev_state == 0:
                                new_state = 1
                                if total_cost < dp[i][c][new_state]:
                                    dp[i][c][new_state] = total_cost
                                    parent[i][c][new_state] = (prev_char, prev_state)
                            elif prev_state == 1:
                                new_state = 2
                                if total_cost < dp[i][c][new_state]:
                                    dp[i][c][new_state] = total_cost
                                    parent[i][c][new_state] = (prev_char, prev_state)
                            else:
                                new_state = 2
                                if total_cost < dp[i][c][new_state]:
                                    dp[i][c][new_state] = total_cost
                                    parent[i][c][new_state] = (prev_char, prev_state)
                        else:
                            if prev_state == 2:
                                new_state = 0
                                if total_cost < dp[i][c][new_state]:
                                    dp[i][c][new_state] = total_cost
                                    parent[i][c][new_state] = (prev_char, prev_state)
        
        min_cost = INF
        for c in range(26):
            if dp[n-1][c][2] < min_cost:
                min_cost = dp[n-1][c][2]
        if min_cost == INF:
            return ""
        
        candidates = []
        for c in range(26):
            if dp[n-1][c][2] == min_cost:
                candidates.append(c)
                
        results = []
        for c in candidates:
            arr = [0] * n
            arr[n-1] = c
            state = 2
            cur_char = c
            for pos in range(n-1, 0, -1):
                prev_char, prev_state = parent[pos][cur_char][state]
                arr[pos-1] = prev_char
                cur_char = prev_char
                state = prev_state
            s = ''.join(chr(ord('a') + x) for x in arr)
            results.append(s)
        
        results.sort()
        return results[0]