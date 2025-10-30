class Solution:
    def lastVisitedIntegers(self, words: List[str]) -> List[int]:
        nums = []
        result = []
        consecutive_prevs = 0
        
        for word in words:
            if word != "prev":
                nums.append(int(word))
                consecutive_prevs = 0
            else:
                consecutive_prevs += 1
                if consecutive_prevs > len(nums):
                    result.append(-1)
                else:
                    reversed_nums = nums[::-1]
                    result.append(reversed_nums[consecutive_prevs - 1])
        
        return result