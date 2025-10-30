n, q = map(int, input().split())
instructions = [tuple(input().split()) for _ in range(q)]
for i in range(q):
    instructions[i] = (instructions[i][0], int(instructions[i][1]))

l = 1
r = 2
total_steps = 0

for h, t in instructions:
    if h == 'L':
        S = l
        O = r
        T = t
    else:
        S = r
        O = l
        T = t
    
    d_cw = (T - S) % n
    d_ccw = (S - T) % n
    
    # Check if O is on clockwise path
    if S <= T:
        on_cw = (O >= S) and (O <= T)
    else:
        on_cw = (O >= S) or (O <= T)
    
    # Check if O is on counter-clockwise path
    if S >= T:
        on_ccw = (O >= T) and (O <= S)
    else:
        on_ccw = (O <= S) or (O >= T)
    
    if d_cw < d_ccw:
        if on_cw:
            steps = d_ccw
        else:
            steps = d_cw
    elif d_ccw < d_cw:
        if on_ccw:
            steps = d_cw
        else:
            steps = d_ccw
    else:  # d_cw == d_ccw
        if on_cw:
            steps = d_ccw
        elif on_ccw:
            steps = d_cw
        else:
            steps = d_cw
    
    total_steps += steps
    
    if h == 'L':
        l = T
    else:
        r = T

print(total_steps)