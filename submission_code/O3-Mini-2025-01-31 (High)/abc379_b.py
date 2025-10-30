def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    if not input_data:
        return
    N = int(input_data[0])
    K = int(input_data[1])
    S = input_data[2]

    count = 0
    result = 0
    for ch in S:
        if ch == 'O':
            count += 1
            if count == K:
                result += 1
                count = 0
        else:
            count = 0

    print(result)

if __name__ == '__main__':
    main()