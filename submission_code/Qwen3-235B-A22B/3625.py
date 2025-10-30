class Solution:
    def canAliceWin(self, n: int) -> bool:
        current = n
        step = 10
        while True:
            if current < step:
                return step % 2 == 1
            current -= step
            step -= 1