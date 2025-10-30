class Solution:
    def maxIncreasingGroups(self, usageLimits: List[int]) -> int:
        n = len(usageLimits)
        groups = []
        
        for i in range(n):
            if not groups or len(groups[-1]) < usageLimits[i]:
                groups.append([i])
            else:
                for j in range(len(groups)):
                    if len(groups[j]) < usageLimits[i]:
                        groups[j].append(i)
                        break
        
        return len(groups)