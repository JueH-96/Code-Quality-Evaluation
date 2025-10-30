n = int(input())
max_palindrome = 0
x = 1
while True:
    cube = x ** 3
    if cube > n:
        break
    if str(cube) == str(cube)[::-1]:
        if cube > max_palindrome:
            max_palindrome = cube
    x += 1
print(max_palindrome)