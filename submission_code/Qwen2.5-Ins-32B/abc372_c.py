import sys

def solve():
    input = sys.stdin.read
    N, Q, S, *queries = input().split()
    N, Q = int(N), int(Q)
    S = list(S)
    queries = list(zip(map(int, queries[::2]), queries[1::2]))
    
    # Initial count of "ABC" substrings
    count = 0
    for i in range(N-2):
        if S[i] == 'A' and S[i+1] == 'B' and S[i+2] == 'C':
            count += 1
    
    results = []
    for X, C in queries:
        X -= 1  # Convert to 0-based index
        # Check if the change affects the count of "ABC"
        if S[X] == 'A' and X+1 < N and S[X+1] == 'B' and X+2 < N and S[X+2] == 'C':
            count -= 1
        if S[X] == 'B' and X-1 >= 0 and S[X-1] == 'A' and X+1 < N and S[X+1] == 'C':
            count -= 1
        if S[X] == 'C' and X-2 >= 0 and S[X-2] == 'A' and X-1 >= 0 and S[X-1] == 'B':
            count -= 1
        
        S[X] = C
        
        # Check if the new character forms a new "ABC"
        if S[X] == 'A' and X+1 < N and S[X+1] == 'B' and X+2 < N and S[X+2] == 'C':
            count += 1
        if S[X] == 'B' and X-1 >= 0 and S[X-1] == 'A' and X+1 < N and S[X+1] == 'C':
            count += 1
        if S[X] == 'C' and X-2 >= 0 and S[X-2] == 'A' and X-1 >= 0 and S[X-1] == 'B':
            count += 1
        
        results.append(str(count))
    
    print('
'.join(results))

solve()