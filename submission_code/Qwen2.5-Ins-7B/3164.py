class Solution:
    def lastVisitedIntegers(self, words: List[str]) -> List[int]:
        last_visited = []
        prev_count = 0
        result = []
        
        for word in words:
            if word != "prev":
                last_visited.append(int(word))
                prev_count = 0
            else:
                prev_count += 1
                if prev_count <= len(last_visited):
                    result.append(last_visited[-prev_count])
                else:
                    result.append(-1)
        
        return result