n = int(input())
max_pal = 0
x = 1
while True:
    cube = x ** 3
    if cube > n:
        break
    s = str(cube)
    if s == s[::-1]:
        max_pal = cube
    x += 1
print(max_pal)