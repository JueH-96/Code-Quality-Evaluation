import sys

def restore_records(S):
    n = len(S)
    if n % 2 == 0:
        return 0
    
    count = 0
    for i in range(n):
        if i % 2 == 0 and S[i] != 'i':
            count += 1
        elif i % 2 == 1 and S[i] != 'o':
            count += 1
    
    return count

# Read input from stdin
S = input().strip()

# Solve the problem
answer = restore_records(S)

# Write output to stdout
print(answer)