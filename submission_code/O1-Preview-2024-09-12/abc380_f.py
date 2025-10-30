# YOUR CODE HERE
import sys
import threading
def main():
    sys.setrecursionlimit(1 << 25)
    N, M, L = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))
    B = list(map(int, sys.stdin.readline().split()))
    C = list(map(int, sys.stdin.readline().split()))
    K = N + M + L
    cards = A + B + C
    value = cards  # value[i] is the number on card i
    pow3 = [1]
    for _ in range(K):
        pow3.append(pow3[-1] * 3)
    # s_i: location of card i
    # 0: Takahashi's hand, 1: Aoki's hand, 2: table
    s0 = 0
    for i in range(K):
        if i < N:
            s_i = 0  # Takahashi's hand
        elif i < N + M:
            s_i = 1  # Aoki's hand
        else:
            s_i = 2  # Table
        s0 += s_i * pow3[i]
    memo = {}
    def dfs(s, turn):
        if (s, turn) in memo:
            return memo[(s, turn)]
        current_player = turn  # 0: Takahashi, 1: Aoki
        opponent = 1 - turn
        has_move = False
        for i in range(K):
            s_i = (s // pow3[i]) % 3
            if s_i == current_player:
                has_move = True
                break
        if not has_move:
            # Cannot make move, current player loses
            memo[(s, turn)] = False
            return False
        # Try all possible moves
        for c in range(K):
            s_c = (s // pow3[c]) % 3
            if s_c != current_player:
                continue
            # Play card c: move c from hand to table
            s_after_play = s + (2 - s_c) * pow3[c]
            # Find table cards with value less than value[c]
            possible_d = [None]  # Option to not take any card
            for d in range(K):
                s_d = (s_after_play // pow3[d]) % 3
                if s_d == 2 and value[d] < value[c]:
                    possible_d.append(d)
            for d in possible_d:
                s_new = s_after_play
                if d is not None:
                    # Move d from table to current player's hand
                    s_d = (s_new // pow3[d]) % 3
                    s_new += (current_player - s_d) * pow3[d]
                # Recurse to opponent's turn
                opponent_wins = dfs(s_new, opponent)
                if not opponent_wins:
                    # Current player wins
                    memo[(s, turn)] = True
                    return True
        # All moves lead to opponent winning
        memo[(s, turn)] = False
        return False
    winner = dfs(s0, 0)
    if winner:
        print("Takahashi")
    else:
        print("Aoki")
threading.Thread(target=main).start()