def main():
    X = int(input().strip())
    total = 2025
    count = 0
    for i in range(1, 10):
        if X % i == 0:
            j = X // i
            if j <= 9:
                count += 1
    result = total - count * X
    print(result)

if __name__ == '__main__':
    main()