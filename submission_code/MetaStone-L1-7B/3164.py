class Solution:
    def lastVisitedIntegers(self, words: List[str]) -> List[int]:
        nums = []
        current_k = 0
        result = []
        for word in words:
            if word == "prev":
                current_k += 1
                k = current_k
                if len(nums) >= k:
                    last = nums[::-1][k-1]
                else:
                    last = -1
                result.append(last)
            else:
                nums.append(int(word))
        return result