class Solution:
    def numberOfAlternatingGroups(self, colors: List[int], queries: List[List[int]]) -> List[int]:
        n = len(colors)
        ans = []

        def count_alternating_groups(size):
            count = 0
            for i in range(n):
                valid = True
                for j in range(size):
                    curr_idx = (i + j) % n
                    if j > 0:
                        prev_idx = (i + j - 1) % n
                        if colors[curr_idx] == colors[prev_idx]:
                            valid = False
                            break
                if valid:
                    count += 1
            return count
        
        for query in queries:
            if query[0] == 1:
                ans.append(count_alternating_groups(query[1]))
            elif query[0] == 2:
                colors[query[1]] = query[2]
        
        return ans