class Solution:
    def maxIncreasingGroups(self, usageLimits: List[int]) -> int:
        n = len(usageLimits)
        count = [0] * n
        groups = 0
        group_size = 0
        
        while True:
            group_size +=1
            available = []
            for i in range(n):
                if count[i] < usageLimits[i]:
                    available.append(i)
            
            if len(available) < group_size:
                break
            
            
            available.sort(key = lambda x: usageLimits[x]-count[x], reverse = True)
            
            for i in range(group_size):
                count[available[i]] +=1
            groups += 1
        return groups