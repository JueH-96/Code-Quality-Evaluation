class Solution:
    def numberOfAlternatingGroups(self, colors: List[int], queries: List[List[int]]) -> List[int]:
        def count_groups(size):
            count = 0
            for i in range(len(colors)):
                if all(colors[j] != colors[j+1] for j in range(i, i+size)):
                    count += 1
            return count

        res = []
        for query in queries:
            if query[0] == 1:
                res.append(count_groups(query[1]))
            else:
                colors[query[1]] = query[2]
        return res