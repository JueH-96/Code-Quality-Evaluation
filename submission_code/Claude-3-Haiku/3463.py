class Solution:
    def numberOfAlternatingGroups(self, colors: List[int]) -> int:
        count = 0
        n = len(colors)
        
        for i in range(n):
            if i == 0 or colors[i] != colors[i-1]:
                for j in range(i+1, n):
                    if colors[j] != colors[i]:
                        for k in range(j+1, n):
                            if colors[k] == colors[i]:
                                count += 1
                                break
                        break
        
        return count