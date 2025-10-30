class Solution:
    def lastVisitedIntegers(self, words):
        nums = []
        res = []
        prev_count = 0
        for word in words:
            if word == "prev":
                prev_count += 1
                if prev_count > len(nums):
                    res.append(-1)
                else:
                    res.append(nums[-prev_count])
            else:
                nums.append(int(word))
                prev_count = 0
        return res