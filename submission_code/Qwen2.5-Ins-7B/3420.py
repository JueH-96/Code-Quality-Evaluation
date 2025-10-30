from typing import List

class Solution:
    def occurrencesOfElement(self, nums: List[int], queries: List[int], x: int) -> List[int]:
        index_map = {}
        for i, num in enumerate(nums):
            if num == x:
                if x not in index_map:
                    index_map[x] = []
                index_map[x].append(i)
        
        result = []
        for query in queries:
            if x in index_map and len(index_map[x]) >= query:
                result.append(index_map[x][query - 1])
            else:
                result.append(-1)
        return result