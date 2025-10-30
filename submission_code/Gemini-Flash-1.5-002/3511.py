class Solution:
    def losingPlayer(self, x: int, y: int) -> str:
        turn = 0
        while True:
            found = False
            for i in range(x + 1):
                rem = 115 - 75 * i
                if rem >= 0 and rem % 10 == 0 and rem // 10 <= y:
                    x -= i
                    y -= rem // 10
                    found = True
                    break
            if not found:
                if turn == 0:
                    return "Bob"
                else:
                    return "Alice"
            turn = 1 - turn