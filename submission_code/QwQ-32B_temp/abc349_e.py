import sys
from functools import lru_cache

def main():
    A = []
    for _ in range(3):
        row = list(map(int, sys.stdin.readline().split()))
        A.append(row)
    A_flat = [
        A[0][0], A[0][1], A[0][2],
        A[1][0], A[1][1], A[1][2],
        A[2][0], A[2][1], A[2][2]
    ]

    def has_win(state, player):
        lines = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],
            [0, 3, 6], [1, 4, 7], [2, 5, 8],
            [0, 4, 8], [2, 4, 6]
        ]
        for line in lines:
            if all(state[i] == player for i in line):
                return True
        return False

    def is_game_over(state):
        return has_win(state, 1) or has_win(state, 2)

    def is_full(state):
        return 0 not in state

    def compute_scores(state):
        tak = 0
        aoki = 0
        for i in range(9):
            if state[i] == 1:
                tak += A_flat[i]
            elif state[i] == 2:
                aoki += A_flat[i]
        return tak, aoki

    @lru_cache(maxsize=None)
    def can_win(state):
        if is_game_over(state):
            winner = None
            if has_win(state, 1):
                winner = 1
            elif has_win(state, 2):
                winner = 2
            else:
                assert False, "Unexpected state"
            moves_count = sum(1 for cell in state if cell != 0)
            current_player = 1 if (moves_count % 2 == 0) else 2
            return (winner == current_player)
        elif is_full(state):
            tak, aoki = compute_scores(state)
            winner = 1 if tak > aoki else 2
            moves_count = sum(1 for cell in state if cell != 0)
            current_player = 1 if (moves_count % 2 == 0) else 2
            return (winner == current_player)
        else:
            moves_count = sum(1 for cell in state if cell != 0)
            current_player = 1 if (moves_count % 2 == 0) else 2
            for i in range(9):
                if state[i] == 0:
                    new_state_list = list(state)
                    new_state_list[i] = current_player
                    new_state = tuple(new_state_list)
                    if has_win(new_state, current_player):
                        return True
                    if not can_win(new_state):
                        return True
            return False

    initial_state = tuple([0] * 9)
    result = can_win(initial_state)
    print("Takahashi" if result else "Aoki")

if __name__ == "__main__":
    main()