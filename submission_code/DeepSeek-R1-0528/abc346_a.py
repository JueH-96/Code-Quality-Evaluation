def main():
    n = int(input().strip())
    arr = list(map(int, input().split()))
    result = []
    for i in range(n - 1):
        product = arr[i] * arr[i + 1]
        result.append(str(product))
    print(" ".join(result))

if __name__ == '__main__':
    main()