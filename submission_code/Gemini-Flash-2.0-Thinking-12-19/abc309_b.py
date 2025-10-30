def solve():
    n = int(input())
    a = []
    for _ in range(n):
        row_str = input()
        row = [int(char) for char in row_str]
        a.append(row)
    
    if n <= 1:
        for row in a:
            print("".join(map(str, row)))
        return
        
    outer_coords = []
    if n == 2:
        outer_coords = [(0,0), (0,1), (1,1), (1,0)]
    else:
        for j in range(n):
            outer_coords.append((0, j))
        for i in range(1, n):
            outer_coords.append((i, n-1))
        for j in range(n-2, -1, -1):
            outer_coords.append((n-1, j))
        for i in range(n-2, 0, -1):
            outer_coords.append((i, 0))
            
    outer_values = []
    for r, c in outer_coords:
        outer_values.append(a[r][c])
        
    shifted_outer_values = [0] * len(outer_values)
    for i in range(len(outer_values)):
        shifted_outer_values[(i + 1) % len(outer_values)] = outer_values[i]
        
    b = [row[:] for row in a]
    
    for i in range(len(outer_coords)):
        r, c = outer_coords[i]
        b[r][c] = shifted_outer_values[i]
        
    for i in range(n):
        print("".join(map(str, b[i])))

if __name__ == '__main__':
    solve()