class Solution:
    def numberOfAlternatingGroups(self, colors: List[int], queries: List[List[int]]) -> List[int]:
        from collections import defaultdict
        
        def count_groups(size):
            count = 0
            i = 0
            while i < len(colors):
                if len(colors[i:i+size]) == size and is_alternating(i, size):
                    count += 1
                i += 1
            return count
        
        def is_alternating(start, size):
            for i in range(size - 1):
                if (colors[(start + i) % len(colors)] + colors[(start + i + 1) % len(colors)]) % 2 == 0:
                    return False
            return True
        
        results = []
        for query in queries:
            if query[0] == 1:
                results.append(count_groups(query[1]))
            elif query[0] == 2:
                colors[query[1]] = query[2]
        return results