class Solution:
    def canAliceWin(self, n: int) -> bool:
        next_take = 10
        alice_turn = True
        current_stones = n
        while True:
            if next_take <= 0 or next_take > current_stones:
                return not alice_turn
            current_stones -= next_take
            alice_turn = not alice_turn
            next_take -= 1