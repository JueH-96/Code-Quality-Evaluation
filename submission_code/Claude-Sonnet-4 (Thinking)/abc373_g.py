def segments_intersect(p1, q1, p2, q2):
    def orientation(p, q, r):
        val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
        if val == 0:
            return 0  # collinear
        return 1 if val > 0 else 2  # clockwise or counterclockwise
    
    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)
    
    # General case
    if o1 != o2 and o3 != o4:
        return True
    
    return False

n = int(input())
P = []
Q = []

for i in range(n):
    a, b = map(int, input().split())
    P.append((a, b))

for i in range(n):
    c, d = map(int, input().split())
    Q.append((c, d))

def backtrack(pos, result, used):
    if pos == n:
        return True
    
    for i in range(n):
        if used[i]:
            continue
        
        # Check if connecting P[pos] to Q[i] creates any intersection
        valid = True
        for j in range(pos):
            if segments_intersect(P[pos], Q[i], P[j], Q[result[j]]):
                valid = False
                break
        
        if valid:
            result[pos] = i
            used[i] = True
            if backtrack(pos + 1, result, used):
                return True
            used[i] = False
    
    return False

result = [0] * n
used = [False] * n

if backtrack(0, result, used):
    # Convert to 1-indexed
    print(' '.join(str(result[i] + 1) for i in range(n)))
else:
    print(-1)