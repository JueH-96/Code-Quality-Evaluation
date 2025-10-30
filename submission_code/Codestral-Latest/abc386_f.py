# YOUR CODE HERE
import sys
from itertools import product

input = sys.stdin.read
data = input().split()

K = int(data[0])
S = data[1]
T = data[2]

def can_transform(S, T, K):
    len_s = len(S)
    len_t = len(T)

    if abs(len_s - len_t) > K:
        return False

    for operations in product(range(4), repeat=K):
        temp_S = list(S)
        for op in operations:
            if op == 0:  # Insert
                temp_S.append('')
            elif op == 1:  # Delete
                if temp_S:
                    temp_S.pop()
            elif op == 2:  # Replace
                if temp_S:
                    temp_S[-1] = ''
            elif op == 3:  # Change
                if temp_S:
                    temp_S[-1] = T[0]
        if ''.join(temp_S) == T:
            return True

    return False

if can_transform(S, T, K):
    print("Yes")
else:
    print("No")