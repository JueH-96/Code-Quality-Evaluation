class Solution:
    def lastVisitedIntegers(self, words: List[str]) -> List[int]:
        nums = []
        result = []
        consecutive_prev_count = 0
        
        for word in words:
            if word == "prev":
                consecutive_prev_count += 1
                k = consecutive_prev_count
                if k <= len(nums):
                    result.append(nums[-k])
                else:
                    result.append(-1)
            else:
                # It's an integer
                nums.append(int(word))
                consecutive_prev_count = 0
        
        return result