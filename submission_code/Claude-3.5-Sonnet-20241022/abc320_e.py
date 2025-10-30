N, M = map(int, input().split())
noodles = [0] * N  # Amount of noodles each person has
absent = [0] * N   # Time until person returns to line (0 means present)

for _ in range(M):
    T, W, S = map(int, input().split())
    
    # Find first person in line
    first_person = -1
    for i in range(N):
        if T >= absent[i]:  # Person is present
            first_person = i
            break
    
    # Give noodles to first person if any
    if first_person != -1:
        noodles[first_person] += W
        absent[first_person] = T + S

for i in range(N):
    print(noodles[i])