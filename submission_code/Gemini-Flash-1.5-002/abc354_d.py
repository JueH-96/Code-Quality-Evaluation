# YOUR CODE HERE
A, B, C, D = map(int, input().split())

def get_color(x, y):
    if (int(x) + int(y/2)) % 2 == 0:
        if int(y) % 2 == 0:
            if (int(x) + int(y)) % 2 == 0:
                return 0
            else:
                return 1
        else:
            if (int(x) + int(y)) % 2 == 0:
                return 1
            else:
                return 0
    else:
        if int(y) % 2 == 0:
            if (int(x) + int(y)) % 2 == 0:
                return 1
            else:
                return 0
        else:
            if (int(x) + int(y)) % 2 == 0:
                return 0
            else:
                return 1

black_area = 0
total_area = (C - A) * (D - B)

for i in range(int(A), int(C)):
    for j in range(int(B), int(D)):
        if get_color(i + 0.5, j + 0.5) == 0:
            black_area += 1

print(int(2 * black_area))