# YOUR CODE HERE
import sys

def has_intersection(N, chords):
    stack = []
    for a, b in chords:
        if a > b:
            a, b = b, a
        while stack and stack[-1][0] > a and stack[-1][1] < b:
            if stack[-1][0] < a and stack[-1][1] > a:
                return True
            stack.pop()
        stack.append((a, b))
    return False

N = int(input())
chords = [tuple(map(int, input().split())) for _ in range(N)]

if has_intersection(N, chords):
    print("Yes")
else:
    print("No")