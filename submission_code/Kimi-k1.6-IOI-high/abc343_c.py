n = int(input())
max_cube = 0
x = 1
while True:
    cube = x ** 3
    if cube > n:
        break
    if str(cube) == str(cube)[::-1]:
        max_cube = cube
    x += 1
print(max_cube)