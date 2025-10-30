class Solution:
    def numberOfAlternatingGroups(self, colors: List[int], queries: List[List[int]]) -> List[int]:
        n = len(colors)
        ans = []
        for query in queries:
            if query[0] == 2:
                colors[query[1]] = query[2]
            else:
                size = query[1]
                count = 0
                for i in range(n):
                    group = []
                    for j in range(size):
                        group.append(colors[(i + j) % n])
                    
                    is_alternating = True
                    for k in range(1, size -1):
                        if group[k] == group[k+1]:
                            is_alternating = False
                            break
                    if is_alternating and len(set(group))==2 and len(group)>0:
                        count +=1

                ans.append(count)
        return ans