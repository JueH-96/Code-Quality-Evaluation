class Solution:
    def lastVisitedIntegers(self, words: List[str]) -> List[int]:
        result = []
        visited_integers = []
        prev_count = 0
        
        for word in words:
            if word == "prev":
                prev_count += 1
                if prev_count <= len(visited_integers):
                    result.append(visited_integers[-prev_count])
                else:
                    result.append(-1)
            else:
                visited_integers.append(int(word))
                prev_count = 0
        
        return result