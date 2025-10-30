class Solution:
    def numberOfAlternatingGroups(self, colors: List[int], queries: List[List[int]]) -> List[int]:
        n = len(colors)
        
        def count_alternating_groups(size):
            count = 0
            i = 0
            while i < n:
                if colors[i] == 0:
                    j = i
                    while j < n and colors[j] == 0:
                        j += 1
                    if j - i == size:
                        count += 1
                    i = j
                else:
                    j = i
                    while j < n and colors[j] == 1:
                        j += 1
                    if j - i == size:
                        count += 1
                    i = j
            return count
        
        result = []
        for query in queries:
            if query[0] == 1:
                result.append(count_alternating_groups(query[1]))
            else:
                colors[query[1]] = query[2]
        
        return result