class Solution:
    def getMaxFunctionValue(self, receiver: List[int], k: int) -> int:
        n = len(receiver)
        max_val = 0
        for x in range(n):
            f_x = 0
            for _ in range(k):
                f_x += receiver[x]
                x = receiver[x]
            max_val = max(max_val, f_x)
        return max_val