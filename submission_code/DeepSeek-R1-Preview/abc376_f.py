n, q = map(int, input().split())
current_l = 1
current_r = 2
total_steps = 0

for _ in range(q):
    h, t = input().split()
    t = int(t)
    if h == 'L':
        pos = current_l
        other = current_r
    else:
        pos = current_r
        other = current_l
    
    d_cw = (t - pos) % n
    d_ccw = (pos - t) % n
    d = min(d_cw, d_ccw)
    
    # Check if other is at the target
    if other == t:
        d += 1
    
    total_steps += d
    
    # Update the current position of the hand
    if h == 'L':
        current_l = t
    else:
        current_r = t

print(total_steps)