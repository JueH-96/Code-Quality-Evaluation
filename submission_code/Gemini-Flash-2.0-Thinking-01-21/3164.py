class Solution:
    def lastVisitedIntegers(self, words: List[str]) -> List[int]:
        nums = []
        result = []
        prev_count = 0
        for word in words:
            if word == "prev":
                prev_count += 1
                k = prev_count
                if k > len(nums) or len(nums) == 0:
                    result.append(-1)
                else:
                    result.append(nums[-k])
            else:
                nums.append(int(word))
                prev_count = 0
        return result