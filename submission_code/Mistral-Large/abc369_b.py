import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
A = []
S = []

for i in range(1, N+1):
    A.append(int(data[2*i-1]))
    S.append(data[2*i])

left_hand_position = A[0]
right_hand_position = A[0]
fatigue = 0

for i in range(N):
    if S[i] == 'L':
        fatigue += abs(A[i] - left_hand_position)
        left_hand_position = A[i]
    else:
        fatigue += abs(A[i] - right_hand_position)
        right_hand_position = A[i]

sys.stdout.write(str(fatigue) + '
')