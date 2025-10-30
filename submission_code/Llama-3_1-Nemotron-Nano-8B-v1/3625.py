class Solution:
    def canAliceWin(self, n: int) -> bool:
        remaining = n
        step = 0
        while True:
            current_take = 10 - step
            if current_take <= 0:
                return step % 2 == 0 and False or True  # Simplified return for clarity
            if remaining < current_take:
                return step % 2 == 0  # Alice's turn if step is even, she loses
            remaining -= current_take
            step += 1