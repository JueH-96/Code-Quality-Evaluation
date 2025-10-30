n, q = map(int, input().split())
l_pos = 1
r_pos = 2
total = 0

for _ in range(q):
    h, t = input().split()
    t = int(t)
    if h == 'L':
        current = l_pos
        other = r_pos
        other_char = 'R'
    else:
        current = r_pos
        other = l_pos
        other_char = 'L'
    
    if current == t:
        continue
    
    a = current
    target = t
    o = other
    
    # Compute d_cw and d_ccw
    if target >= a:
        d_cw = target - a
    else:
        d_cw = target - a + n
    d_ccw = n - d_cw
    short_dist = min(d_cw, d_ccw)
    
    if o == target:
        # Need to move other away
        new_o = target + 1 if target < n else 1
        total += 1
        # Add short_dist steps to move current to target
        total += short_dist
        # Update other's position
        if other_char == 'R':
            r_pos = new_o
        else:
            l_pos = new_o
        # Update current's position
        if h == 'L':
            l_pos = target
        else:
            r_pos = target
    else:
        on_short = False
        if d_cw < d_ccw:
            # Check if o is on clockwise path from a to target
            if a < target:
                on_short = a < o <= target
            else:
                on_short = o > a or o <= target
        elif d_ccw < d_cw:
            # Check if o is on clockwise path from target to a (counter-clockwise path from a to target)
            if target < a:
                on_short = target < o <= a
            else:
                on_short = o > target or o <= a
        else:
            # Equal distance, check clockwise path from a to target
            if a < target:
                on_short = a < o <= target
            else:
                on_short = o > a or o <= target
        
        if on_short:
            option1 = max(d_cw, d_ccw)
            option2 = 1 + short_dist
            if option1 < option2:
                total += option1
            else:
                total += option2
                # Move other away
                new_o = target + 1 if target < n else 1
                if other_char == 'R':
                    r_pos = new_o
                else:
                    l_pos = new_o
        else:
            total += short_dist
        
        # Update current's position
        if h == 'L':
            l_pos = target
        else:
            r_pos = target

print(total)