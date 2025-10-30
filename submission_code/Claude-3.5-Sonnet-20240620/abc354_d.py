# YOUR CODE HERE
def calculate_black_area(a, b, c, d):
    def count_black(x1, y1, x2, y2):
        if x1 > x2 or y1 > y2:
            return 0
        
        total = (x2 - x1) * (y2 - y1)
        even_x = (x2 // 2) - ((x1 - 1) // 2)
        even_y = (y2 // 2) - ((y1 - 1) // 2)
        even_sum = ((x2 + y2) // 2) - ((x1 + y1 - 1) // 2)
        
        black = total - even_x - even_y - even_sum
        if (x1 + y1) % 2 == 0:
            black = total - black
        
        return black

    area = count_black(a, b, c, d)
    return 2 * area

a, b, c, d = map(int, input().split())
result = calculate_black_area(a, b, c, d)
print(result)