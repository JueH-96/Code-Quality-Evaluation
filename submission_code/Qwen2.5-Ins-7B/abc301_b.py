# YOUR CODE HERE
from collections import deque

def insert_sequence(A):
    dq = deque(A)
    while True:
        for i in range(len(dq) - 1):
            if abs(dq[i] - dq[i + 1]) == 1:
                continue
            if dq[i] < dq[i + 1]:
                for j in range(dq[i] + 1, dq[i + 1]):
                    dq.insert(i + 1, j)
                break
            else:
                for j in range(dq[i] - 1, dq[i + 1], -1):
                    dq.insert(i + 1, j)
                break
        else:
            break
    return list(dq)

N = int(input())
A = list(map(int, input().split()))
result = insert_sequence(A)
print(" ".join(map(str, result)))