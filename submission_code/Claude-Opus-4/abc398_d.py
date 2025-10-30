N, R, C = map(int, input().split())
S = input().strip()

# Set to store cells with smoke
smoke = {(0, 0)}
result = []

for i in range(N):
    # Apply wind to move all smoke
    new_smoke = set()
    for r, c in smoke:
        if S[i] == 'N':
            new_smoke.add((r - 1, c))
        elif S[i] == 'W':
            new_smoke.add((r, c - 1))
        elif S[i] == 'S':
            new_smoke.add((r + 1, c))
        elif S[i] == 'E':
            new_smoke.add((r, c + 1))
    
    smoke = new_smoke
    
    # If (0,0) is empty, add new smoke there
    if (0, 0) not in smoke:
        smoke.add((0, 0))
    
    # Check if smoke exists at (R, C) at time i+0.5
    if (R, C) in smoke:
        result.append('1')
    else:
        result.append('0')

print(''.join(result))