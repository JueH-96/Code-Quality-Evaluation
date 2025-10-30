# Read input
N, Q = map(int, input().split())
left, right = 1, 2
total_steps = 0

for _ in range(Q):
    H, T = input().split()
    T = int(T)
    
    if H == 'L':
        A, B, C = left, T, right
    else:
        A, B, C = right, T, left
    
    d_clockwise = (B - A) % N
    d_AC = (C - A) % N
    d_CB = (B - C) % N
    
    if d_AC + d_CB == d_clockwise:
        steps = N - d_clockwise
    else:
        steps = d_clockwise
    
    total_steps += steps
    
    if H == 'L':
        left = B
    else:
        right = B

print(total_steps)