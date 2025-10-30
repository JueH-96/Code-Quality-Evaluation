from sortedcontainers import SortedList

class Solution:
    def resultsArray(self, queries: List[List[int]], k: int) -> List[int]:
        ob = SortedList()
        res = []
        
        for i, (x, y) in enumerate(queries):
            ob.add(abs(x) + abs(y))
            
            if len(ob) < k:
                res.append(-1)
            else:
                res.append(ob[k-1])
            
        return res