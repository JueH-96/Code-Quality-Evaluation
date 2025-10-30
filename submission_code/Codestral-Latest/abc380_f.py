def can_win(hand, table):
    # Check if there is any card in the table that is less than any card in the hand
    for card in hand:
        if any(card > t for t in table):
            return True
    return False

def game_outcome(N, M, L, A, B, C):
    # Sort the cards in descending order
    A.sort(reverse=True)
    B.sort(reverse=True)
    C.sort(reverse=True)

    # Check if Takahashi can win
    if can_win(A, C):
        return "Takahashi"

    # Check if Aoki can win
    if can_win(B, C + A):
        return "Aoki"

    return "Aoki"

# Read input
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
M = int(data[1])
L = int(data[2])

A = list(map(int, data[3:3+N]))
B = list(map(int, data[3+N:3+N+M]))
C = list(map(int, data[3+N+M:3+N+M+L]))

# Determine the outcome
result = game_outcome(N, M, L, A, B, C)

# Print the result
print(result)