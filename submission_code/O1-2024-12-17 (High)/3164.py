from typing import List

class Solution:
    def lastVisitedIntegers(self, words: List[str]) -> List[int]:
        visited = []
        results = []
        consecutive_prev_count = 0
        
        for w in words:
            if w == "prev":
                consecutive_prev_count += 1
                # Last visited integer index from the end
                if consecutive_prev_count <= len(visited):
                    results.append(visited[-consecutive_prev_count])
                else:
                    results.append(-1)
            else:
                # Reset consecutive "prev" count and store integer
                visited.append(int(w))
                consecutive_prev_count = 0
        
        return results