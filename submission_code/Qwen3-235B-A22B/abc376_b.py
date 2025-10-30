def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    Q = int(input[idx])
    idx += 1
    
    current_left = 1
    current_right = 2
    total = 0
    
    for _ in range(Q):
        H = input[idx]
        idx += 1
        T = int(input[idx])
        idx += 1
        
        if H == 'L':
            moving = current_left
            other_pos = current_right
        else:
            moving = current_right
            other_pos = current_left
        
        if moving == T:
            continue  # No steps needed
        
        S = moving
        F = other_pos
        target = T
        
        # Determine allowed directions
        # Check clockwise blocked
        cw_blocked = False
        if S < target:
            if F > S and F <= target:
                cw_blocked = True
        else:
            if F > S or F <= target:
                cw_blocked = True
        allowed_cw = not cw_blocked
        
        # Check counter-clockwise blocked
        cc_blocked = False
        if S < target:
            if F <= S or F >= target:
                cc_blocked = True
        else:
            if target <= F <= S:
                cc_blocked = True
        allowed_cc = not cc_blocked
        
        # Collect possible steps
        candidates = []
        if allowed_cw:
            cw_steps = (target - S) % N
            candidates.append(cw_steps)
        if allowed_cc:
            cc_steps = (S - target) % N
            candidates.append(cc_steps)
        
        steps = min(candidates)
        total += steps
        
        # Update position
        if H == 'L':
            current_left = target
        else:
            current_right = target
    
    print(total)

if __name__ == '__main__':
    main()