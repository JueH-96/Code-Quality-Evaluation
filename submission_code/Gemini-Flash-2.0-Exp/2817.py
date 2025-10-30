class Solution:
    def minimumCost(self, s: str) -> int:
        n = len(s)
        ans = float('inf')

        # Case 1: Make all characters '0'
        cost1 = 0
        temp_s = list(s)
        for i in range(n - 1):
            if temp_s[i] != '0':
                cost1 += min(i + 1, n - i)
                for j in range(i + 1):
                    temp_s[j] = '1' if temp_s[j] == '0' else '0'
        if temp_s[n - 1] != '0':
            cost1 += 1
        ans = min(ans, cost1)

        # Case 2: Make all characters '1'
        cost2 = 0
        temp_s = list(s)
        for i in range(n - 1):
            if temp_s[i] != '1':
                cost2 += min(i + 1, n - i)
                for j in range(i + 1):
                    temp_s[j] = '1' if temp_s[j] == '0' else '0'
        if temp_s[n - 1] != '1':
            cost2 += 1
        ans = min(ans, cost2)

        return ans