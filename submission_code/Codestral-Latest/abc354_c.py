import sys
from collections import defaultdict

input = sys.stdin.read
data = input().split()

N = int(data[0])
cards = []
for i in range(N):
    A_i = int(data[2 * i + 1])
    C_i = int(data[2 * i + 2])
    cards.append((A_i, C_i, i + 1))

cards.sort()

stack = []
for card in cards:
    while stack and stack[-1][1] > card[1]:
        stack.pop()
    stack.append(card)

remaining_cards = [card[2] for card in stack]
remaining_cards.sort()

print(len(remaining_cards))
print(" ".join(map(str, remaining_cards)))