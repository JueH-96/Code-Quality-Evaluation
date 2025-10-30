class Solution:
    def lastVisitedIntegers(self, words: List[str]) -> List[int]:
        ans = []
        nums = []
        consec_prev = 0
        for word in words:
            if word == "prev":
                consec_prev += 1
                k = consec_prev
                reversed_nums = nums[::-1]
                if k > len(reversed_nums):
                    ans.append(-1)
                else:
                    ans.append(reversed_nums[k - 1])
            else:
                consec_prev = 0
                nums.append(int(word))
        return ans