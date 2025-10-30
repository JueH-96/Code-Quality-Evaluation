A, B, C, D = map(int, input().split())

def mod2(x):
    return x % 2

width = C - A
height = D - B

term = mod2(A) + mod2(B) + mod2(C) + mod2(D) + mod2(width) * mod2(height)
ans = width * height + term // 2

print(ans * 2)