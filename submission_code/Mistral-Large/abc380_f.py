import sys
from itertools import product

def can_take(card, table):
    return any(c < card for c in table)

def play_game(takahashi, aoki, table, turn=True):
    if not takahashi and not aoki:
        return not turn

    if turn:
        for card in takahashi:
            new_takahashi = takahashi - {card}
            if can_take(card, table):
                new_table = (table | {card}) - {min(c for c in table if c < card)}
            else:
                new_table = table | {card}
            if not play_game(new_takahashi, aoki, new_table, not turn):
                return True
        return False
    else:
        for card in aoki:
            new_aoki = aoki - {card}
            if can_take(card, table):
                new_table = (table | {card}) - {min(c for c in table if c < card)}
            else:
                new_table = table | {card}
            if not play_game(takahashi, new_aoki, new_table, not turn):
                return True
        return False

def main():
    input = sys.stdin.read
    data = input().split()
    N, M, L = map(int, data[:3])
    A = set(map(int, data[3:3+N]))
    B = set(map(int, data[3+N:3+N+M]))
    C = set(map(int, data[3+N+M:3+N+M+L]))

    if play_game(A, B, C):
        print("Takahashi")
    else:
        print("Aoki")

if __name__ == "__main__":
    main()