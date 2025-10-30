def main():
    N = int(input().strip())
    ans = 1
    for x in range(1, 1000001):
        cube = x * x * x
        if cube > N:
            break
        s = str(cube)
        if s == s[::-1]:
            ans = cube
    print(ans)

if __name__ == '__main__':
    main()