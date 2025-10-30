class Solution:
    def lastVisitedIntegers(self, words: List[str]) -> List[int]:
        visited_integers = []  # To store the visited integers in order
        result = []            # To store results for each "prev"
        prev_streak = 0        # To count consecutive "prev"s

        for word in words:
            if word == "prev":
                prev_streak += 1
                # The index we want in visited_integers is the one from the end
                # i.e. if prev_streak == 1, we want the last integer; if prev_streak == 2, we want the second to last integer, etc.
                if prev_streak <= len(visited_integers):
                    result.append(visited_integers[-prev_streak])
                else:
                    result.append(-1)
            else:
                # We encountered an integer, reset the consecutive "prev" streak
                prev_streak = 0
                visited_integers.append(int(word))
        
        return result