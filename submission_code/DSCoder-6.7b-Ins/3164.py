class Solution:
    def lastVisitedIntegers(self, words: List[str]) -> List[int]:
        nums = []
        prev_count = 0
        result = []
        for word in words:
            if word == "prev":
                prev_count += 1
                if len(nums) >= prev_count:
                    result.append(nums[-prev_count])
                else:
                    result.append(-1)
            else:
                nums.append(int(word))
                result.append(int(word))
        return result