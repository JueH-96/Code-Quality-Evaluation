class Solution:
    def lastVisitedIntegers(self, words: List[str]) -> List[int]:
        nums = []
        consecutive_prev = 0
        result = []
        
        for word in words:
            if word == "prev":
                consecutive_prev += 1
                k = consecutive_prev
                reversed_nums = nums[::-1]
                if k > len(reversed_nums):
                    result.append(-1)
                else:
                    result.append(reversed_nums[k-1])
            else:
                consecutive_prev = 0
                nums.append(int(word))
        
        return result