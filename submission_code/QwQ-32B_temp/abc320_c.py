M = int(input())
S1 = input().strip()
S2 = input().strip()
S3 = input().strip()

min_time = float('inf')

for c in '0123456789':
    if c not in S1 or c not in S2 or c not in S3:
        continue
    
    positions1 = [i for i, ch in enumerate(S1) if ch == c]
    positions2 = [i for i, ch in enumerate(S2) if ch == c]
    positions3 = [i for i, ch in enumerate(S3) if ch == c]
    
    for p1 in positions1:
        for p2 in positions2:
            for p3 in positions3:
                a, b, c_p = sorted([p1, p2, p3])
                
                if a < b < c_p:
                    current = c_p
                elif a == b and b < c_p:
                    current = max(c_p, a + M)
                elif a < b and b == c_p:
                    current = b + M
                else:  # all equal
                    current = a + 2 * M
                
                if current < min_time:
                    min_time = current

if min_time == float('inf'):
    print(-1)
else:
    print(min_time)