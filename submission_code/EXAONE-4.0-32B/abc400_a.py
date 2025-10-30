def main():
    A = int(input().strip())
    if 400 % A == 0:
        B = 400 // A
        print(B)
    else:
        print(-1)

if __name__ == '__main__':
    main()