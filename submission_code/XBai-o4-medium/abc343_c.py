def main():
    import sys
    N = int(sys.stdin.readline().strip())
    max_pal = 0
    x = 1
    while True:
        cube = x ** 3
        if cube > N:
            break
        s = str(cube)
        if s == s[::-1]:
            if cube > max_pal:
                max_pal = cube
        x += 1
    print(max_pal)

if __name__ == "__main__":
    main()