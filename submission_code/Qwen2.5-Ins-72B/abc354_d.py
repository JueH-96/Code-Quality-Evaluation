# YOUR CODE HERE
import sys

def calculate_black_area(A, B, C, D):
    def is_black(x, y):
        return (x + y) % 2 == 0

    def area_of_rectangle(A, B, C, D):
        return (C - A) * (D - B)

    def area_of_black_in_rectangle(A, B, C, D):
        if A % 2 == 0:
            black_width = (C - A) // 2
            if (C - A) % 2 == 1:
                black_width += 1
        else:
            black_width = (C - A) // 2

        if B % 2 == 0:
            black_height = (D - B) // 2
            if (D - B) % 2 == 1:
                black_height += 1
        else:
            black_height = (D - B) // 2

        total_black_area = black_width * (D - B) + black_height * (C - A) - black_width * black_height
        return total_black_area

    total_area = area_of_rectangle(A, B, C, D)
    black_area = area_of_black_in_rectangle(A, B, C, D)
    return 2 * black_area

A, B, C, D = map(int, sys.stdin.readline().strip().split())
print(calculate_black_area(A, B, C, D))