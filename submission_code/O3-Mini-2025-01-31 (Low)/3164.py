from typing import List

class Solution:
    def lastVisitedIntegers(self, words: List[str]) -> List[int]:
        result = []
        visited = []   # List to store the integers we've seen so far.
        consecutive_prev = 0  # Counter for consecutive "prev" occurrences.
        
        for word in words:
            if word != "prev":
                # Reset the counter for consecutive "prev" when we see an integer.
                consecutive_prev = 0
                # Convert the string to integer and add it to visited.
                visited.append(int(word))
            else:
                # When we see "prev", increment the counter.
                consecutive_prev += 1
                # Calculate the index in the reversed list.
                if consecutive_prev <= len(visited):
                    # Using -consecutive_prev gives the k-th element from end.
                    result.append(visited[-consecutive_prev])
                else:
                    result.append(-1)
                    
        return result