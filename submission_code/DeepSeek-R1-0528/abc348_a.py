def main():
    N = int(input().strip())
    res = []
    for i in range(1, N + 1):
        if i % 3 == 0:
            res.append('x')
        else:
            res.append('o')
    print(''.join(res))

if __name__ == "__main__":
    main()