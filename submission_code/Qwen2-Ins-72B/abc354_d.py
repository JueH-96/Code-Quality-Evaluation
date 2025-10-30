A, B, C, D = map(int, input().split())

def f(x, y):
    return (x + y) % 2

def g(x, y):
    return (x + y) % 4

def h(x, y):
    return (x + y) % 8

def solve(A, B, C, D):
    if g(A, B) == 0:
        if h(C, B) == 0:
            if f(C, B):
                x1 = (C - A) // 2 * 2 + 1
            else:
                x1 = (C - A) // 2 * 2
        else:
            x1 = (C - A) // 2 * 2
        if h(A, D) == 0:
            if f(A, D):
                y1 = (D - B) // 2 * 2 + 1
            else:
                y1 = (D - B) // 2 * 2
        else:
            y1 = (D - B) // 2 * 2
    elif g(A, B) == 2:
        if h(C, B) == 2:
            if f(C, B):
                x1 = (C - A) // 2 * 2 + 1
            else:
                x1 = (C - A) // 2 * 2
        else:
            x1 = (C - A) // 2 * 2
        if h(A, D) == 2:
            if f(A, D):
                y1 = (D - B) // 2 * 2 + 1
            else:
                y1 = (D - B) // 2 * 2
        else:
            y1 = (D - B) // 2 * 2
    elif g(A, B) == 4:
        if h(C, B) == 4:
            if f(C, B):
                x1 = (C - A) // 2 * 2 + 1
            else:
                x1 = (C - A) // 2 * 2
        else:
            x1 = (C - A) // 2 * 2
        if h(A, D) == 4:
            if f(A, D):
                y1 = (D - B) // 2 * 2 + 1
            else:
                y1 = (D - B) // 2 * 2
        else:
            y1 = (D - B) // 2 * 2
    else:
        if h(C, B) == 6:
            if f(C, B):
                x1 = (C - A) // 2 * 2 + 1
            else:
                x1 = (C - A) // 2 * 2
        else:
            x1 = (C - A) // 2 * 2
        if h(A, D) == 6:
            if f(A, D):
                y1 = (D - B) // 2 * 2 + 1
            else:
                y1 = (D - B) // 2 * 2
        else:
            y1 = (D - B) // 2 * 2
    return x1 * y1

print(solve(A, B, C, D))