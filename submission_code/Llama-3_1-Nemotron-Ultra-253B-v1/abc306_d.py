n = int(input())
courses = [tuple(map(int, input().split())) for _ in range(n)]

prev_h = 0
prev_u = float('-inf')

for x, y in courses:
    # Calculate new_healthy
    candidates_h = [prev_h]
    if x == 0:
        candidates_h.append(prev_h + y)
        candidates_h.append(prev_u + y)
    new_h = max(candidates_h)
    
    # Calculate new_upset
    candidates_u = [prev_u]
    if x == 1:
        candidates_u.append(prev_h + y)
    new_u = max(candidates_u)
    
    prev_h, prev_u = new_h, new_u

print(max(prev_h, prev_u))