n, q = map(int, input().split())
current_l = 1
current_r = 2
total = 0

for _ in range(q):
    h, t = input().split()
    t = int(t)
    if h == 'L':
        a = current_l
        b = current_r
    else:
        a = current_r
        b = current_l
    new_pos = t
    if a == new_pos:
        steps = 0
    else:
        # Compute clockwise
        d_clockwise = (new_pos - a) % n
        if a < new_pos:
            in_clockwise = (b > a and b < new_pos)
        else:
            in_clockwise = (b > a or b < new_pos)
        
        # Compute counter-clockwise
        d_counter = (a - new_pos) % n
        if new_pos < a:
            in_counter = (b > new_pos and b < a)
        else:
            in_counter = (b < a or b > new_pos)
        
        candidates = []
        if not in_clockwise:
            candidates.append(d_clockwise)
        if not in_counter:
            candidates.append(d_counter)
        steps = min(candidates)
    total += steps
    if h == 'L':
        current_l = new_pos
    else:
        current_r = new_pos

print(total)