from typing import List

class Solution:
    def lastVisitedIntegers(self, words: List[str]) -> List[int]:
        visited = []  # List to store the visited integers
        result = []   # List for the result for each "prev"
        consecutive_prev = 0  # To count consecutive "prev" occurrences
        
        for word in words:
            if word == "prev":
                # Increase the consecutive "prev" count
                consecutive_prev += 1
                # If the count is more than visited integers, answer is -1
                if consecutive_prev <= len(visited):
                    # The last visited integer is the (consecutive_prev-1)th element in reverse order.
                    answer = visited[-consecutive_prev]
                else:
                    answer = -1
                result.append(answer)
            else:
                # When a number is encountered, add it to visited and reset the consecutive "prev" counter
                visited.append(int(word))
                consecutive_prev = 0
        return result