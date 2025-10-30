def is_right_triangle(x1, y1, x2, y2, x3, y3):
    a = (x2 - x1)**2 + (y2 - y1)**2
    b = (x3 - x2)**2 + (y3 - y2)**2
    c = (x3 - x1)**2 + (y3 - y1)**2
    if a == b + c or b == a + c or c == a + b:
        return 'Yes'
    else:
        return 'No'

x1, y1 = map(int, input().split())
x2, y2 = map(int, input().split())
x3, y3 = map(int, input().split())

print(is_right_triangle(x1, y1, x2, y2, x3, y3))