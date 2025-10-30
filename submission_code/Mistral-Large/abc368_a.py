import sys

input = sys.stdin.read
data = input().split()

N = int(data[0])
K = int(data[1])
cards = list(map(int, data[2:]))

# Take the last K cards and move them to the front
cards = cards[:-K] + cards[-K:]

# Print the result
print(" ".join(map(str, cards)))