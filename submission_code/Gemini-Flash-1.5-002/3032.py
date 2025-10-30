class Solution:
    def getMaxFunctionValue(self, receiver: List[int], k: int) -> int:
        n = len(receiver)
        max_f = 0
        for x in range(n):
            f_x = x
            current_player = x
            for _ in range(k):
                current_player = receiver[current_player]
                f_x += current_player
            max_f = max(max_f, f_x)
        return max_f