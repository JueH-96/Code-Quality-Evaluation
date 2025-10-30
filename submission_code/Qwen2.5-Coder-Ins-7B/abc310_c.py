# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
sticks = data[1:]

stick_set = set()

for stick in sticks:
    if stick in stick_set or stick[::-1] in stick_set:
        continue
    stick_set.add(stick)

print(len(stick_set))