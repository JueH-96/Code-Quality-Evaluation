import sys

def is_right_triangle(x_A, y_A, x_B, y_B, x_C, y_C):
    def distance_squared(x1, y1, x2, y2):
        return (x1 - x2) ** 2 + (y1 - y2) ** 2

    AB_sq = distance_squared(x_A, y_A, x_B, y_B)
    AC_sq = distance_squared(x_A, y_A, x_C, y_C)
    BC_sq = distance_squared(x_B, y_B, x_C, y_C)

    sides_sq = sorted([AB_sq, AC_sq, BC_sq])

    return sides_sq[0] + sides_sq[1] == sides_sq[2]

def main():
    input = sys.stdin.read
    data = input().split()

    x_A, y_A = map(int, data[0:2])
    x_B, y_B = map(int, data[2:4])
    x_C, y_C = map(int, data[4:6])

    if is_right_triangle(x_A, y_A, x_B, y_B, x_C, y_C):
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    main()