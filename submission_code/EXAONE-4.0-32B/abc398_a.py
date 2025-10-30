def main():
    n = int(input().strip())
    arr = ['-'] * n
    if n % 2 == 1:
        mid = n // 2
        arr[mid] = '='
    else:
        mid1 = n // 2 - 1
        mid2 = n // 2
        arr[mid1] = '='
        arr[mid2] = '='
    print(''.join(arr))

if __name__ == '__main__':
    main()