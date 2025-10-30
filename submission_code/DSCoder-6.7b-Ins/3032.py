class Solution:
    def getMaxFunctionValue(self, receiver: List[int], k: int) -> int:
        n = len(receiver)
        max_val = 0
        for x in range(n):
            curr_val = x
            curr_receiver = x
            for _ in range(k):
                curr_receiver = receiver[curr_receiver]
                curr_val += curr_receiver
            max_val = max(max_val, curr_val)
        return max_val