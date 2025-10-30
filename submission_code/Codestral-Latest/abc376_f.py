import sys
input = sys.stdin.read
data = input().split()

n = int(data[0])
q = int(data[1])
instructions = [(data[2*i+2], int(data[2*i+3])) for i in range(q)]

left_hand = 1
right_hand = 2
total_moves = 0

for hand, target in instructions:
    if hand == 'L':
        if left_hand == target:
            continue
        if right_hand == target:
            right_hand = (right_hand + 1) % n or n
            total_moves += 1
        left_hand = target
    else:
        if right_hand == target:
            continue
        if left_hand == target:
            left_hand = (left_hand + 1) % n or n
            total_moves += 1
        right_hand = target

    left_dist = min(abs(left_hand - target), n - abs(left_hand - target))
    right_dist = min(abs(right_hand - target), n - abs(right_hand - target))

    total_moves += left_dist + right_dist

print(total_moves)