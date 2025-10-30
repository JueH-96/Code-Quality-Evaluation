# YOUR CODE HERE
import sys
input = sys.stdin.read

data = input().split()
N = int(data[0])
P = list(map(int, data[1:]))

max_other_score = max(P[1:])
person_1_score = P[0]

if person_1_score > max_other_score:
    print(0)
else:
    print(max_other_score - person_1_score + 1)