class Solution:
    def lastVisitedIntegers(self, words: List[str]) -> List[int]:
        nums = []
        consec_prev = 0
        result = []
        for word in words:
            if word == "prev":
                consec_prev += 1
                k = consec_prev
                if k > len(nums):
                    result.append(-1)
                else:
                    index = len(nums) - k
                    result.append(nums[index])
            else:
                num = int(word)
                nums.append(num)
                consec_prev = 0
        return result