# YOUR CODE HERE
from itertools import permutations

def can_transform(s1, s2):
    diff = sum(1 for a, b in zip(s1, s2) if a != b)
    return diff == 1

def check_sequence(strings):
    for i in range(len(strings) - 1):
        if not can_transform(strings[i], strings[i+1]):
            return False
    return True

N, M = map(int, input().split())
strings = [input().strip() for _ in range(N)]

for perm in permutations(strings):
    if check_sequence(perm):
        print("Yes")
        break
else:
    print("No")