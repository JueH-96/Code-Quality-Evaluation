# YOUR CODE HERE
import sys
input = sys.stdin.read

def can_shout_love_for_takoyaki(A, B, C):
    if B < C:
        if A >= B and A < C:
            return "No"
        else:
            return "Yes"
    else:
        if A >= B or A < C:
            return "No"
        else:
            return "Yes"

data = input().strip().split()
A = int(data[0])
B = int(data[1])
C = int(data[2])

print(can_shout_love_for_takoyaki(A, B, C))