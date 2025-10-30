class Solution:
    def lastVisitedIntegers(self, words: List[str]) -> List[int]:
        nums = []
        for word in words:
            if word == "prev":
                if nums:
                    nums.append(nums[-1] + 1)
                else:
                    nums.append(1)
            else:
                nums.append(int(word))
        return nums[::-1]