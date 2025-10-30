class Solution:
    def lastVisitedIntegers(self, words: List[str]) -> List[int]:
        nums = []
        res = []
        for w in words:
            if w == "prev":
                res.append(-1 if not nums else nums.pop())
            else:
                nums.append(int(w))
        return [r if r == -1 else nums.pop() for r in res]