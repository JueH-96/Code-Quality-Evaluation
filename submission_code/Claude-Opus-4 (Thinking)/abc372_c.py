N, Q = map(int, input().split())
S = list(input().strip())

# Function to check if "ABC" starts at position i
def is_abc_at(S, i):
    if i >= 0 and i + 2 < len(S):
        return S[i] == 'A' and S[i+1] == 'B' and S[i+2] == 'C'
    return False

# Initial count of "ABC"
count = 0
for i in range(N):
    if is_abc_at(S, i):
        count += 1

# Process queries
for _ in range(Q):
    X, C = input().split()
    X = int(X) - 1  # Convert to 0-indexed
    
    # Check positions that could be affected before the change
    # ABC could start at X-2, X-1, or X
    for start in range(X - 2, X + 1):
        if is_abc_at(S, start):
            count -= 1
    
    # Make the change
    S[X] = C
    
    # Check the same positions after the change
    for start in range(X - 2, X + 1):
        if is_abc_at(S, start):
            count += 1
    
    print(count)