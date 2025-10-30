class Solution:
    def lastVisitedIntegers(self, words: List[str]) -> List[int]:
        nums = []
        consec_prev = 0
        result = []
        
        for word in words:
            if word != "prev":
                nums.append(int(word))
                consec_prev = 0
            else:
                consec_prev += 1
                if consec_prev <= len(nums):
                    result.append(nums[-consec_prev])
                else:
                    result.append(-1)
        
        return result