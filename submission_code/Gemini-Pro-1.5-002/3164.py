class Solution:
    def lastVisitedIntegers(self, words: List[str]) -> List[int]:
        nums = []
        res = []
        k = 0
        for word in words:
            if word == "prev":
                k += 1
                if k > len(nums):
                    res.append(-1)
                else:
                    res.append(nums[-k])
            else:
                nums.append(int(word))
                k = 0
        return res