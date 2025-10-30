# YOUR CODE HERE

import sys

N = int(sys.stdin.readline().strip())
S = sys.stdin.readline().strip()

count_A = 0
count_B = 0
count_C = 0

for i in range(N):
    if S[i] == 'A':
        count_A += 1
    elif S[i] == 'B':
        count_B += 1
    elif S[i] == 'C':
        count_C += 1
    if count_A > 0 and count_B > 0 and count_C > 0:
        print(i + 1)
        break