# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
A = []
S = []
for i in range(N):
    A.append(int(data[2*i+1]))
    S.append(data[2*i+2])

left_hand = 0
right_hand = 0
fatigue = 0

for i in range(N):
    if S[i] == 'L':
        if left_hand == 0:
            left_hand = A[i]
        else:
            fatigue += abs(A[i] - left_hand)
            left_hand = A[i]
    else:
        if right_hand == 0:
            right_hand = A[i]
        else:
            fatigue += abs(A[i] - right_hand)
            right_hand = A[i]

print(fatigue)