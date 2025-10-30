def main():
    X = int(input().strip())
    n = 1
    current = 1
    while current < X:
        n += 1
        current *= n
    print(n)

if __name__ == '__main__':
    main()