# YOUR CODE HERE
N, Q = map(int, input().split())

instructions = []
for _ in range(Q):
    H_i, T_i = input().split()
    T_i = int(T_i)
    instructions.append((H_i, T_i))

left_pos = 1
right_pos = 2
total_steps = 0

for H_i, T_i in instructions:
    if H_i == 'L':
        curr_pos = left_pos
        other_pos = right_pos
    else:
        curr_pos = right_pos
        other_pos = left_pos

    target = T_i

    # Compute minimal steps from curr_pos to target avoiding other_pos
    # Try both directions
    min_steps = None

    # Clockwise direction
    steps = 0
    pos = curr_pos
    invalid = False
    while True:
        pos += 1
        if pos > N:
            pos = 1
        steps += 1
        if pos == other_pos:
            invalid = True
            break
        if pos == target:
            break
    if not invalid:
        min_steps = steps

    # Counterclockwise direction
    steps = 0
    pos = curr_pos
    invalid = False
    while True:
        pos -= 1
        if pos < 1:
            pos = N
        steps += 1
        if pos == other_pos:
            invalid = True
            break
        if pos == target:
            break
    if not invalid:
        if min_steps is None or steps < min_steps:
            min_steps = steps

    total_steps += min_steps

    # Update positions
    if H_i == 'L':
        left_pos = target
    else:
        right_pos = target

print(total_steps)