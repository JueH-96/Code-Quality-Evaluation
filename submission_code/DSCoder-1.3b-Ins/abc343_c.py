import sys

def find_max_palindromic_cube(n):
    max_cube = -1
    for i in range(n, 0, -1):
        cube = i ** 3
        if str(cube) == str(cube)[::-1]:
            max_cube = max(max_cube, cube)
    return max_cube

def main():
    n = int(sys.stdin.readline().strip())
    print(find_max_palindromic_cube(n))

if __name__ == "__main__":
    main()