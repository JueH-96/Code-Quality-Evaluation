import sys
def readints():
    return list(map(int, sys.stdin.read().split()))
A, B, C, D = readints()

def get_black_area(A, B, C, D):
    def count(x, y):
        return ((x // 2) * (y // 2)) + (((x + 1) // 2) * ((y + 1) // 2))
    total_area_twice = 2 * (C - A) * (D - B)
    # Assuming half the area is black
    return total_area_twice // 2

print(get_black_area(A, B, C, D))