class Solution:
    def canAliceWin(self, n: int) -> bool:
        stones = n
        k = 1
        while True:
            r = 11 - k
            if r <= 0 or r > stones:
                if k % 2 == 1:  # Alice's turn
                    return False  # Alice loses
                else:  # Bob's turn
                    return True  # Bob loses, Alice wins
            stones -= r
            k += 1