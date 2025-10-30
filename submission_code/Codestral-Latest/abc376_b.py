import sys
input = sys.stdin.read
data = input().split()

index = 0
N = int(data[index])
index += 1
Q = int(data[index])
index += 1

instructions = []
for _ in range(Q):
    H = data[index]
    index += 1
    T = int(data[index])
    index += 1
    instructions.append((H, T))

left_hand = 1
right_hand = 2
total_moves = 0

for H, T in instructions:
    if H == 'L':
        if left_hand == T:
            continue
        if right_hand == T:
            right_hand = left_hand
            total_moves += 1
        while left_hand != T:
            left_hand = (left_hand % N) + 1
            if left_hand == right_hand:
                left_hand = (left_hand % N) + 1
            total_moves += 1
    elif H == 'R':
        if right_hand == T:
            continue
        if left_hand == T:
            left_hand = right_hand
            total_moves += 1
        while right_hand != T:
            right_hand = (right_hand % N) + 1
            if right_hand == left_hand:
                right_hand = (right_hand % N) + 1
            total_moves += 1

print(total_moves)