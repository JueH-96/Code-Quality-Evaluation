class Solution:
    def lastVisitedIntegers(self, words: List[str]) -> List[int]:
        nums = []
        res = []
        k = 0
        for word in words:
            if word == "prev":
                k += 1
                reversed_nums = nums[::-1]
                if k - 1 < len(reversed_nums):
                    res.append(reversed_nums[k - 1])
                else:
                    res.append(-1)
            else:
                nums.append(int(word))
                k = 0
        return res