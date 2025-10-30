class Solution:
    def minimizeStringValue(self, s: str) -> str:
        cost = [0]*26
        for i in range(len(s)-1, -1, -1):
            for j in range(ord(s[i])-ord('a'), 0):
                cost[j] += 1
            cost[ord(s[i])-ord('a')] = 0
            for j in range(0, len(s)-i-1):
                cost[(ord(s[j])-ord('a')+1)%26] += 1
        return min(cost)