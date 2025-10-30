class Solution:
    def lastVisitedIntegers(self, words: List[str]) -> List[int]:
        result = []
        visited_integers = []
        consecutive_prev = 0
        
        for w in words:
            if w == "prev":
                consecutive_prev += 1
                if consecutive_prev > len(visited_integers):
                    result.append(-1)
                else:
                    result.append(visited_integers[-consecutive_prev])
            else:
                consecutive_prev = 0
                visited_integers.append(int(w))
                
        return result