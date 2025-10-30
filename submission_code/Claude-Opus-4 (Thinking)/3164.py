class Solution:
    def lastVisitedIntegers(self, words: List[str]) -> List[int]:
        nums = []
        result = []
        k = 0
        
        for word in words:
            if word != "prev":
                nums.append(int(word))
                k = 0
            else:
                k += 1
                nums_reverse = nums[::-1]
                if k - 1 < len(nums_reverse):
                    result.append(nums_reverse[k - 1])
                else:
                    result.append(-1)
        
        return result