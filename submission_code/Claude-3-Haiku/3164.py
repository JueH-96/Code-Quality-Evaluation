class Solution:
    def lastVisitedIntegers(self, words: List[str]) -> List[int]:
        result = []
        nums = []
        for word in words:
            if word == "prev":
                k = len(result)
                if k > len(nums):
                    result.append(-1)
                else:
                    result.append(nums[-k])
            else:
                nums.append(int(word))
        return result