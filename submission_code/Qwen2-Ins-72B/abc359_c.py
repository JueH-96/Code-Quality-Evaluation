sx, sy = map(int, input().split())
tx, ty = map(int, intp)
sx -= sy % 2
tx -= ty % 2
sy //= 2
ty //= 2
print(abs(sx - tx) + abs(sy - ty))