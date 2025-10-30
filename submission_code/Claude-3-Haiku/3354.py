class Solution:
    def minimizeStringValue(self, s: str) -> str:
        n = len(s)
        cost = [0] * n
        
        # Calculate the cost for each index
        for i in range(n):
            if s[i] != '?':
                for j in range(i):
                    if s[j] == s[i]:
                        cost[i] += 1
        
        # Replace '?' with the lexicographically smallest letter
        result = []
        for i in range(n):
            if s[i] == '?':
                min_cost = float('inf')
                min_char = 'a'
                for c in 'abc':
                    temp_cost = cost[i]
                    for j in range(i):
                        if s[j] == c:
                            temp_cost += 1
                    if temp_cost < min_cost:
                        min_cost = temp_cost
                        min_char = c
                    elif temp_cost == min_cost and c < min_char:
                        min_char = c
                result.append(min_char)
            else:
                result.append(s[i])
        
        return ''.join(result)