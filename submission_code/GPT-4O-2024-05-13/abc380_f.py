# YOUR CODE HERE
import sys
from functools import lru_cache

input = sys.stdin.read
data = input().split()

N = int(data[0])
M = int(data[1])
L = int(data[2])

A = tuple(map(int, data[3:3+N]))
B = tuple(map(int, data[3+N:3+N+M]))
C = tuple(map(int, data[3+N+M:]))

@lru_cache(None)
def game_state(takahashi_hand, aoki_hand, table_cards, turn):
    if turn == 0:  # Takahashi's turn
        for i in range(len(takahashi_hand)):
            new_t_hand = takahashi_hand[:i] + takahashi_hand[i+1:]
            new_table = table_cards + (takahashi_hand[i],)
            if any(card < takahashi_hand[i] for card in table_cards):
                for j in range(len(table_cards)):
                    if table_cards[j] < takahashi_hand[i]:
                        new_t_hand = new_t_hand + (table_cards[j],)
                        new_table = table_cards[:j] + table_cards[j+1:] + (takahashi_hand[i],)
                        break
            if not game_state(new_t_hand, aoki_hand, new_table, 1):
                return True
        return False
    else:  # Aoki's turn
        for i in range(len(aoki_hand)):
            new_a_hand = aoki_hand[:i] + aoki_hand[i+1:]
            new_table = table_cards + (aoki_hand[i],)
            if any(card < aoki_hand[i] for card in table_cards):
                for j in range(len(table_cards)):
                    if table_cards[j] < aoki_hand[i]:
                        new_a_hand = new_a_hand + (table_cards[j],)
                        new_table = table_cards[:j] + table_cards[j+1:] + (aoki_hand[i],)
                        break
            if not game_state(takahashi_hand, new_a_hand, new_table, 0):
                return True
        return False

if game_state(A, B, C, 0):
    print("Takahashi")
else:
    print("Aoki")