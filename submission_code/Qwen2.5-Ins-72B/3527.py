from typing import List

class Solution:
    def numberOfAlternatingGroups(self, colors: List[int], queries: List[List[int]]) -> List[int]:
        def count_alternating_groups(size):
            count = 0
            for start in range(len(colors)):
                valid = True
                for i in range(size):
                    if colors[(start + i) % len(colors)] != (colors[start] + i) % 2:
                        valid = False
                        break
                if valid:
                    count += 1
            return count

        answer = []
        for query in queries:
            if query[0] == 1:
                answer.append(count_alternating_groups(query[1]))
            elif query[0] == 2:
                colors[query[1]] = query[2]
        
        return answer