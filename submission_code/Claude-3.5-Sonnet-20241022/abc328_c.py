N, Q = map(int, input().split())
S = input()

for _ in range(Q):
    l, r = map(int, input().split())
    # Convert to 0-based indexing
    l -= 1
    r -= 1
    
    # Count consecutive same letters in the range
    count = 0
    for i in range(l, r):
        if S[i] == S[i+1]:
            count += 1
            
    print(count)