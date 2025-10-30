class Solution:
    def getMaxFunctionValue(self, receiver: List[int], k: int) -> int:
        n = len(receiver)
        max_val = [0] * n
        for i in range(n):
            cur = i
            total = cur
            for _ in range(min(k, n)):
                cur = receiver[cur]
                total += cur
            max_val[i] = total
        return max(max_val)