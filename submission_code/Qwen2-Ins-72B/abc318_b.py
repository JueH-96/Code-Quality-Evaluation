from sortedcontainers import SortedList

def calculate_area(N, sheets):
    x_coords = SortedList()
    y_coords = SortedList()
    for sheet in sheets:
        x_coords.add(sheet[0])
        x_coords.add(sheet[1])
        y_coords.add(sheet[2])
        y_coords.add(sheet[3])
    
    x_unique = list(x_coords)
    y_unique = list(y_coords)
    
    x_map = {x: i for i, x in enumerate(x_unique)}
    y_map = {y: i for i, y in enumerate(y_unique)}
    
    grid = [[0] * len(y_unique) for _ in range(len(x_unique))]
    
    for sheet in sheets:
        x1, x2, y1, y2 = sheet
        i1, i2 = x_map[x1], x_map[x2]
        j1, j2 = y_map[y1], y_map[y2]
        for i in range(i1, i2):
            for j in range(j1, j2):
                grid[i][j] += 1
    
    area = 0
    for i in range(len(x_unique) - 1):
        for j in range(len(y_unique) - 1):
            if grid[i][j] > 0:
                area += (x_unique[i + 1] - x_unique[i]) * (y_unique[j + 1] - y_unique[j])
    
    return area

N = int(input())
sheets = [list(map(int, input().split())) for _ in range(N)]

print(calculate_area(N, sheets))