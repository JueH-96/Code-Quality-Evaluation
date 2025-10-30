def get_min_transformations(S, T):
    if S == T:
        return [0]
    
    n = len(S)
    # Find all positions where characters differ
    diff_pos = []
    for i in range(n):
        if S[i] != T[i]:
            diff_pos.append(i)
            
    # Try all possible sequences of transformations
    min_steps = []
    min_len = float('inf')
    
    def try_sequence(curr_s, remaining_pos, steps):
        nonlocal min_steps, min_len
        
        if len(steps) > min_len:
            return
            
        if curr_s == T:
            if len(steps) < min_len:
                min_steps = steps[:]
                min_len = len(steps)
            elif len(steps) == min_len and steps < min_steps:
                min_steps = steps[:]
            return
            
        for pos in remaining_pos:
            new_s = list(curr_s)
            new_s[pos] = T[pos]
            new_s = ''.join(new_s)
            
            new_remaining = remaining_pos[:]
            new_remaining.remove(pos)
            
            try_sequence(new_s, new_remaining, steps + [new_s])
    
    try_sequence(S, diff_pos, [])
    return [len(min_steps)] + min_steps

# Read input
S = input().strip()
T = input().strip()

# Get and print result
result = get_min_transformations(S, T)
for line in result:
    print(line)