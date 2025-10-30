def main():
    n, m = map(int, input().split())
    total = 0
    term = 1
    for i in range(m + 1):
        total += term
        if total > 1000000000:
            print('inf')
            return
        if i < m:
            term *= n
    print(total)

if __name__ == '__main__':
    main()