class Solution:
    def lastVisitedIntegers(self, words: List[str]) -> List[int]:
        nums = []
        result = []
        consecutive_prev_count = 0
        
        for word in words:
            if word == "prev":
                consecutive_prev_count += 1
                if consecutive_prev_count <= len(nums):
                    result.append(nums[-consecutive_prev_count])
                else:
                    result.append(-1)
            else:
                nums.append(int(word))
                consecutive_prev_count = 0
        
        return result