def main():
    N = int(input().strip())
    res = []
    for i in range(N + 1):
        char_assigned = False
        for j in range(1, 10):
            if N % j == 0:
                d = N // j
                if i % d == 0:
                    res.append(str(j))
                    char_assigned = True
                    break
        if not char_assigned:
            res.append('-')
    print(''.join(res))

if __name__ == '__main__':
    main()