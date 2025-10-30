# YOUR CODE HERE
def solve():
    a, b, c, d = map(int, input().split())

    def get_color(x, y):
        x_floor = x // 1
        y_floor = y // 1
        
        if (x_floor + y_floor) % 2 == 0:
            return 1
        else:
            return 0

    def calc_black_area(x1, y1, x2, y2):
        total_area = (x2 - x1) * (y2 - y1)
        black_area = 0
        for x in range(int(x1), int(x2)):
            for y in range(int(y1), int(y2)):
                if get_color(x + 0.5, y + 0.5) == 1:
                    black_area +=1
        
        
        
        
        return black_area
    
    print(2 * calc_black_area(a,b,c,d))


solve()