# YOUR CODE HERE
def count_different_sticks(N, sticks):
    unique_sticks = set()
    for stick in sticks:
        normalized_stick = min(stick, stick[::-1])
        unique_sticks.add(normalized_stick)
    return len(unique_sticks)

import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
sticks = data[1:]

print(count_different_sticks(N, sticks))