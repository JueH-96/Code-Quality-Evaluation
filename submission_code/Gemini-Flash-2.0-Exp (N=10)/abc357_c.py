def solve():
    n = int(input())
    size = 3**n
    carpet = [['.' for _ in range(size)] for _ in range(size)]

    def draw_carpet(level, row_start, col_start):
        if level == 0:
            carpet[row_start][col_start] = '#'
            return
        
        block_size = 3**(level - 1)
        
        for i in range(3):
            for j in range(3):
                if i == 1 and j == 1:
                    continue
                else:
                    draw_carpet(level - 1, row_start + i * block_size, col_start + j * block_size)
    
    draw_carpet(n, 0, 0)
    
    for row in carpet:
        print("".join(row))

solve()