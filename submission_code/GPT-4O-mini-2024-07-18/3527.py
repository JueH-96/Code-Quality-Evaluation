class Solution:
    def numberOfAlternatingGroups(self, colors: List[int], queries: List[List[int]]) -> List[int]:
        def count_alternating_groups(size):
            count = 0
            n = len(colors)
            if size > n:
                return 0
            
            # Check the circular arrangement
            for i in range(n):
                if colors[i] != colors[(i + 1) % n]:
                    count += 1
            
            # Each alternating group of size `size` can be formed from two adjacent tiles
            return count // 2 if size % 2 == 0 else count // 2 + (1 if count % 2 == 1 else 0)

        results = []
        for query in queries:
            if query[0] == 1:
                results.append(count_alternating_groups(query[1]))
            elif query[0] == 2:
                colors[query[1]] = query[2]
        
        return results