# YOUR CODE HERE
def solve():
    n = int(input())
    grid = [input() for _ in range(n)]
    
    count = 0
    for i1 in range(n):
        for j1 in range(n):
            if grid[i1][j1] == 'x':
                continue
            for i2 in range(n):
                for j2 in range(n):
                    if grid[i2][j2] == 'x' or (i1 == i2 and j1 == j2):
                        continue
                    for i3 in range(n):
                        for j3 in range(n):
                            if grid[i3][j3] == 'x' or (i1 == i3 and j1 == j3) or (i2 == i3 and j2 == j3):
                                continue
                                
                            if (i1 == i2) + (i1 == i3) + (i2 == i3) == 1 and (j1 == j2) + (j1 == j3) + (j2 == j3) == 1:
                                
                                if (i1 == i2 and j1 != j2 and i1 != i3 and j1 != j3 and i2 != i3 and j2 != j3) or \
                                   (i1 == i3 and j1 != j3 and i1 != i2 and j1 != j2 and i3 != i2 and j3 != j2) or \
                                   (i2 == i3 and j2 != j3 and i2 != i1 and j2 != j1 and i3 != i1 and j3 != j1):
                                    
                                    
                                    rows = sorted([i1, i2, i3])
                                    cols = sorted([j1, j2, j3])
                                    
                                    count +=1
                                    
    print(count // 2)

solve()