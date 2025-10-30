A, B, C, D = map(int, input().split())
width = C - A
height = D - B

def solve(A, B, C, D):
    w = C - A
    h = D - B
    return w * h + ((w % 2) * (h % 2))

area = solve(A, B, C, D)
print(area)