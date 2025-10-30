def main():
    import sys
    input = sys.stdin.readline
    X = int(input().strip())
    total = 0
    # Iterate over 9-by-9 multiplication table (i from 1 to 9, j from 1 to 9)
    for i in range(1, 10):
        for j in range(1, 10):
            value = i * j
            if value != X:
                total += value
    print(total)

if __name__ == '__main__':
    main()