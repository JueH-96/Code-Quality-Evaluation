class Solution:
    def numberOfAlternatingGroups(self, colors: List[int], queries: List[List[int]]) -> List[int]:
        n = len(colors)
        ans = []

        def count_alternating_groups(size):
            count = 0
            for i in range(n):
                valid = True
                for j in range(size):
                    if j > 0 and colors[(i + j) % n] == colors[(i + j - 1) % n]:
                        valid = False
                        break
                if valid:
                    count += 1
            return count

        for query in queries:
            if query[0] == 1:
                ans.append(count_alternating_groups(query[1]))
            else:
                colors[query[1]] = query[2]
        return ans