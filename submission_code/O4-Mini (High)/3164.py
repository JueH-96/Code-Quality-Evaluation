class Solution:
    def lastVisitedIntegers(self, words: List[str]) -> List[int]:
        visited_nums = []
        results = []
        prev_count = 0
        
        for w in words:
            if w == "prev":
                prev_count += 1
                # If we have seen at least prev_count integers, grab the kth last one:
                if prev_count <= len(visited_nums):
                    results.append(visited_nums[-prev_count])
                else:
                    results.append(-1)
            else:
                # It's a new integer: record it and reset consecutive-prev count
                visited_nums.append(int(w))
                prev_count = 0
        
        return results