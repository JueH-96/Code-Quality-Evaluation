class Solution:
    def makeTheIntegerZero(self, num1: int, num2: int) -> int:
        q = [(num1, 0)]
        visited = {num1}
        while q:
            curr_num, steps = q.pop(0)
            if curr_num == 0:
                return steps
            for i in range(61):
                next_num = curr_num - (2**i + num2)
                if 0 <= next_num <= 10**9 and next_num not in visited:
                    q.append((next_num, steps + 1))
                    visited.add(next_num)
        return -1