def main():
    n = int(input().strip())
    c1 = (n // 5) * 5
    c2 = c1 + 5
    if c2 > 100:
        print(c1)
    else:
        d1 = abs(n - c1)
        d2 = abs(n - c2)
        if d1 <= d2:
            print(c1)
        else:
            print(c2)

if __name__ == '__main__':
    main()