def main():
    N = int(input().strip())
    total_length = 2 * N + 1
    s = []
    for i in range(total_length):
        if i % 2 == 0:
            s.append('1')
        else:
            s.append('0')
    print(''.join(s))

if __name__ == '__main__':
    main()