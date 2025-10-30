# YOUR CODE HERE

import sys

def solve():
    n = int(input())
    cards = [list(map(int, input().split())) for _ in range(n)]

    # Create a dictionary to store the count of each number on the front and back of the cards
    count = {}
    for a, b in cards:
        if a not in count:
            count[a] = [0, 0]
        if b not in count:
            count[b] = [0, 0]
        count[a][0] += 1
        count[b][1] += 1

    # Check if there is a number that appears on both sides of the cards
    for k, v in count.items():
        if v[0] > 0 and v[1] > 0:
            return "Aoki"

    return "Takahashi"

print(solve())