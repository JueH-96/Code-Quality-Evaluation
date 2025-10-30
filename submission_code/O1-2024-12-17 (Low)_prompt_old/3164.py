class Solution:
    def lastVisitedIntegers(self, words: List[str]) -> List[int]:
        visited = []  # Store all visited integers
        result = []   # Store the answer for each "prev"
        consecutive_prev_count = 0

        for w in words:
            if w == "prev":
                consecutive_prev_count += 1
                # We look for the integer visited[-consecutive_prev_count] if possible
                if consecutive_prev_count <= len(visited):
                    result.append(visited[-consecutive_prev_count])
                else:
                    result.append(-1)
            else:
                # Reset consecutive "prev" count and store the integer
                visited.append(int(w))
                consecutive_prev_count = 0

        return result