def main():
    n = int(input().strip())
    cnt = 0
    while (n & 1) == 0:
        cnt += 1
        n >>= 1
    print(cnt)

if __name__ == '__main__':
    main()