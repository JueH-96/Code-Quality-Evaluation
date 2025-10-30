class Solution:
    def lastVisitedIntegers(self, words: List[str]) -> List[int]:
        nums = []
        res = []
        i = 0
        while i < len(words):
            if words[i] != "prev":
                nums.append(int(words[i]))
            else:
                # Count k: number of consecutive "prev" strings including current
                k = 0
                j = i
                while j >= 0 and words[j] == "prev":
                    k += 1
                    j -= 1
                if k > len(nums):
                    res.append(-1)
                else:
                    last_visited = nums[-k]
                    res.append(last_visited)
            i += 1
        return res