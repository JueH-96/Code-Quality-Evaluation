from typing import List

class Solution:
    def lastVisitedIntegers(self, words: List[str]) -> List[int]:
        result = []
        visited = []
        # counter for consecutive "prev"
        consec_prev = 0
        
        for w in words:
            if w != "prev":
                # reset prev counter and record integer
                consec_prev = 0
                visited.append(int(w))
            else:
                # it's a "prev"
                consec_prev += 1
                k = consec_prev
                # if we have at least k visited integers, take the k-th from the end
                if k <= len(visited):
                    result.append(visited[-k])
                else:
                    result.append(-1)
        return result

# Example usage:
if __name__ == "__main__":
    sol = Solution()
    print(sol.lastVisitedIntegers(["1","2","prev","prev","prev"]))  # [2, 1, -1]
    print(sol.lastVisitedIntegers(["1","prev","2","prev","prev"]))  # [1, 2, 1]