class Solution:
    def lastVisitedIntegers(self, words: List[str]) -> List[int]:
        result = []
        integers = []
        consecutive_prevs = 0
        
        for word in words:
            if word == "prev":
                consecutive_prevs += 1
                if consecutive_prevs > len(integers):
                    result.append(-1)
                else:
                    result.append(integers[-consecutive_prevs])
            else:
                integers.append(int(word))
                consecutive_prevs = 0
        
        return result