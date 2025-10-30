from collections import defaultdict

N, M, Sx, Sy = map(int, input().split())

horizontal_houses = defaultdict(list)  # y -> list of x
vertical_houses = defaultdict(list)    # x -> list of y

for _ in range(N):
    x, y = map(int, input().split())
    horizontal_houses[y].append(x)
    vertical_houses[x].append(y)

visited_houses = set()
curr_x, curr_y = Sx, Sy

for _ in range(M):
    d, c = input().split()
    c = int(c)
    
    if d == 'U':
        # Move from (curr_x, curr_y) to (curr_x, curr_y + c)
        for hy in vertical_houses[curr_x]:
            if curr_y <= hy <= curr_y + c:
                visited_houses.add((curr_x, hy))
        curr_y += c
    elif d == 'D':
        # Move from (curr_x, curr_y) to (curr_x, curr_y - c)
        for hy in vertical_houses[curr_x]:
            if curr_y - c <= hy <= curr_y:
                visited_houses.add((curr_x, hy))
        curr_y -= c
    elif d == 'L':
        # Move from (curr_x, curr_y) to (curr_x - c, curr_y)
        for hx in horizontal_houses[curr_y]:
            if curr_x - c <= hx <= curr_x:
                visited_houses.add((hx, curr_y))
        curr_x -= c
    elif d == 'R':
        # Move from (curr_x, curr_y) to (curr_x + c, curr_y)
        for hx in horizontal_houses[curr_y]:
            if curr_x <= hx <= curr_x + c:
                visited_houses.add((hx, curr_y))
        curr_x += c

print(curr_x, curr_y, len(visited_houses))