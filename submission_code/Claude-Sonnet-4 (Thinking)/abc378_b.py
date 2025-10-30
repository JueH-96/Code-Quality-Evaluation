N = int(input())
garbage_info = []
for i in range(N):
    q, r = map(int, input().split())
    garbage_info.append((q, r))

Q = int(input())
for j in range(Q):
    t, d = map(int, input().split())
    t -= 1  # Convert to 0-indexed
    q, r = garbage_info[t]
    
    # Find next collection day
    next_day = d + ((r - d % q + q) % q)
    print(next_day)