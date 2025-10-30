class Solution:
    def lastVisitedIntegers(self, words: List[str]) -> List[int]:
        result = []
        nums = []
        consecutive_prev = 0
        
        for word in words:
            if word == "prev":
                consecutive_prev += 1
                if consecutive_prev > len(nums):
                    result.append(-1)
                else:
                    result.append(nums[-consecutive_prev])
            else:
                nums.append(int(word))
                consecutive_prev = 0
        
        return result