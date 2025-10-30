def check_line(p1, p2, p3, p4, p5):
    # Check if 5 points lie on same line with equal intervals
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4
    x5, y5 = p5
    
    # Check if points are equally spaced
    dx1 = x2 - x1
    dy1 = y2 - y1
    
    if (x3 != x2 + dx1 or y3 != y2 + dy1 or
        x4 != x3 + dx1 or y4 != y3 + dy1 or
        x5 != x4 + dx1 or y5 != y4 + dy1):
        return False
        
    return True

def is_adjacent(p1, p2):
    # Check if two points are adjacent (share side or corner)
    x1, y1 = p1
    x2, y2 = p2
    return abs(x1-x2) <= 1 and abs(y1-y2) <= 1

H, W = map(int, input().split())
grid = []
for _ in range(H):
    grid.append(input())

# Find all positions of each letter
s_pos = []
n_pos = []
u_pos = []
k_pos = []
e_pos = []

for i in range(H):
    for j in range(W):
        if grid[i][j] == 's':
            s_pos.append((i+1, j+1))
        elif grid[i][j] == 'n':
            n_pos.append((i+1, j+1))
        elif grid[i][j] == 'u':
            u_pos.append((i+1, j+1))
        elif grid[i][j] == 'k':
            k_pos.append((i+1, j+1))
        elif grid[i][j] == 'e':
            e_pos.append((i+1, j+1))

# Try all possible combinations
for s in s_pos:
    for n in n_pos:
        if not is_adjacent(s, n):
            continue
        for u in u_pos:
            if not is_adjacent(n, u):
                continue
            for k in k_pos:
                if not is_adjacent(u, k):
                    continue
                for e in e_pos:
                    if not is_adjacent(k, e):
                        continue
                    if check_line(s, n, u, k, e):
                        print(s[0], s[1])
                        print(n[0], n[1])
                        print(u[0], u[1])
                        print(k[0], k[1])
                        print(e[0], e[1])
                        exit()