class Solution:
    def lastVisitedIntegers(self, words: List[str]) -> List[int]:
        nums = []
        result = []
        consecutive_prev_count = 0
        
        for word in words:
            if word.isdigit():
                nums.append(int(word))
                consecutive_prev_count = 0
            elif word == "prev":
                if consecutive_prev_count < len(nums):
                    result.append(nums[-consecutive_prev_count - 1])
                else:
                    result.append(-1)
                consecutive_prev_count += 1
        
        return result