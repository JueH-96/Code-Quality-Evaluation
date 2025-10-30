class Solution:
    def lastVisitedIntegers(self, words: List[str]) -> List[int]:
        nums = []
        result = []
        consecutive_prev = 0
        for word in words:
            if word == "prev":
                consecutive_prev += 1
                k = consecutive_prev
                if len(nums) < k:
                    result.append(-1)
                else:
                    result.append(nums[-k])
            else:
                nums.append(int(word))
                consecutive_prev = 0
        return result