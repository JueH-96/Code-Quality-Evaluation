class Solution:
    def lastVisitedIntegers(self, words: List[str]) -> List[int]:
        nums = []
        prev_count = 0
        result = []
        
        for word in words:
            if word == "prev":
                prev_count += 1
                if prev_count > len(nums):
                    result.append(-1)
                else:
                    result.append(nums[-prev_count])
            else:
                nums.append(int(word))
                prev_count = 0
        
        return result