def solve():
    A, B, C, D = map(int, input().split())
    
    width = C - A
    height = D - B
    total_area = width * height
    
    white_area = 0.0
    
    for x in range(A, C):
        for y in range(B, D):
            if is_black(x + 0.5, y + 0.5):
                continue
            else:
                white_area += 1
    
    black_area = total_area - white_area
    
    print(int(2 * black_area))

def is_black(x, y):
    
    x_int = int(x)
    y_int = int(y)
    
    
    if (x_int + y_int) % 2 == 0:
        if (x_int % 2 == 0 and y_int % 2 == 0) or (x_int % 2 != 0 and y_int % 2 != 0):
            return True
        else:
            return False
    else:
        if (x_int % 2 == 0 and y_int % 2 != 0) or (x_int % 2 != 0 and y_int % 2 == 0):
            return True
        else:
            return False

solve()