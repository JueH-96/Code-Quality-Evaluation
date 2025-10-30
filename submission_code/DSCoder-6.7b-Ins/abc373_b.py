# YOUR CODE HERE

import sys

def min_distance(s):
    keyboard = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    pos = {char: i+1 for i, char in enumerate(s)}
    return sum(abs(pos[keyboard[i]] - pos[keyboard[i+1]]) for i in range(len(keyboard) - 1))

s = sys.stdin.readline().strip()
print(min_distance(s))