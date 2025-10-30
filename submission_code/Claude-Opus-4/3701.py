class Solution:
    def minCostGoodCaption(self, caption: str) -> str:
        n = len(caption)
        if n < 3:
            return ""
        
        # For each position i and character c, calculate the minimum cost
        # to convert caption[0:i+1] to a good caption ending with character c
        INF = float('inf')
        
        # dp[i][c] = (min_cost, resulting_string) to convert caption[0:i+1] 
        # to a good caption where the last group has character c
        dp = {}
        
        # Initialize for positions 0, 1, 2 (minimum group size is 3)
        for end in range(2, n):
            for c in range(26):
                char = chr(ord('a') + c)
                # Try to make caption[0:end+1] all the same character
                cost = 0
                possible = True
                
                for i in range(end + 1):
                    diff = abs(ord(caption[i]) - ord(char))
                    if diff > 1:
                        possible = False
                        break
                    cost += diff
                
                if possible and end >= 2:
                    dp[(end, c)] = (cost, char * (end + 1))
        
        # Fill dp for remaining positions
        for end in range(3, n):
            for c in range(26):
                char = chr(ord('a') + c)
                
                # Option 1: Extend the previous group
                if (end - 1, c) in dp:
                    prev_cost, prev_str = dp[(end - 1, c)]
                    diff = abs(ord(caption[end]) - ord(char))
                    if diff <= 1:
                        new_cost = prev_cost + diff
                        new_str = prev_str + char
                        if (end, c) not in dp or new_cost < dp[(end, c)][0]:
                            dp[(end, c)] = (new_cost, new_str)
                        elif new_cost == dp[(end, c)][0] and new_str < dp[(end, c)][1]:
                            dp[(end, c)] = (new_cost, new_str)
                
                # Option 2: Start a new group of at least size 3
                for group_start in range(max(0, end - 10), end - 1):
                    if end - group_start + 1 < 3:
                        continue
                    
                    # Check if we can convert caption[group_start:end+1] to char
                    cost = 0
                    possible = True
                    for i in range(group_start, end + 1):
                        diff = abs(ord(caption[i]) - ord(char))
                        if diff > 1:
                            possible = False
                            break
                        cost += diff
                    
                    if not possible:
                        continue
                    
                    # Check all possible previous characters
                    if group_start == 0:
                        # This is the first group
                        if (end, c) not in dp or cost < dp[(end, c)][0]:
                            dp[(end, c)] = (cost, char * (end + 1))
                        elif cost == dp[(end, c)][0]:
                            new_str = char * (end + 1)
                            if new_str < dp[(end, c)][1]:
                                dp[(end, c)] = (cost, new_str)
                    else:
                        for prev_c in range(26):
                            if prev_c == c:  # Can't have same character in adjacent groups
                                continue
                            if (group_start - 1, prev_c) in dp:
                                prev_cost, prev_str = dp[(group_start - 1, prev_c)]
                                total_cost = prev_cost + cost
                                new_str = prev_str + char * (end - group_start + 1)
                                
                                if (end, c) not in dp or total_cost < dp[(end, c)][0]:
                                    dp[(end, c)] = (total_cost, new_str)
                                elif total_cost == dp[(end, c)][0] and new_str < dp[(end, c)][1]:
                                    dp[(end, c)] = (total_cost, new_str)
        
        # Find the best solution
        best_cost = INF
        best_str = ""
        
        for c in range(26):
            if (n - 1, c) in dp:
                cost, result_str = dp[(n - 1, c)]
                if cost < best_cost:
                    best_cost = cost
                    best_str = result_str
                elif cost == best_cost and result_str < best_str:
                    best_str = result_str
        
        return best_str if best_cost < INF else ""