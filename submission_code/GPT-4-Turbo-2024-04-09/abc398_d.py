import sys
input = sys.stdin.read

def solve():
    data = input().split()
    N = int(data[0])
    R = int(data[1])
    C = int(data[2])
    S = data[3]
    
    # Initial position of the smoke
    smoke_r, smoke_c = 0, 0
    
    # Result string to be built
    result = []
    
    # Process each time step
    for i in range(N):
        # Check if the smoke is at Takahashi's position
        if smoke_r == R and smoke_c == C:
            result.append('1')
        else:
            result.append('0')
        
        # Move the smoke according to the wind direction
        if S[i] == 'N':
            smoke_r -= 1
        elif S[i] == 'W':
            smoke_c -= 1
        elif S[i] == 'S':
            smoke_r += 1
        elif S[i] == 'E':
            smoke_c += 1
    
    # Final check for the last position
    if smoke_r == R and smoke_c == C:
        result.append('1')
    else:
        result.append('0')
    
    # We need to output the result for times 1.5 to N+0.5, which corresponds to indices 1 to N in the result list
    print(''.join(result[1:]))