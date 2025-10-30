class Solution:
    def lastVisitedIntegers(self, words: List[str]) -> List[int]:
        nums = []
        consec_prev = 0
        result = []
        for word in words:
            if word == "prev":
                consec_prev += 1
                k = consec_prev
                reversed_nums = nums[::-1]
                if k <= len(reversed_nums):
                    result.append(reversed_nums[k-1])
                else:
                    result.append(-1)
            else:
                nums.append(int(word))
                consec_prev = 0
        return result