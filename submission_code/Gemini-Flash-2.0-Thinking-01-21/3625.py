class Solution:
    def canAliceWin(self, n: int) -> bool:
        if n < 10:
            return False

        def play_game(stones, remove_count, is_alice_turn):
            if stones < remove_count:
                return not is_alice_turn
            
            next_stones = stones - remove_count
            next_remove_count = remove_count - 1
            if next_remove_count < 1:
                return is_alice_turn # The other player cannot make a move, so current player wins

            return not play_game(next_stones, next_remove_count, not is_alice_turn)

        return play_game(n - 10, 9, False)