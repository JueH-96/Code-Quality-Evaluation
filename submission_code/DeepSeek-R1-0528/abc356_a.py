def main():
    N, L, R = map(int, input().split())
    arr = list(range(1, N + 1))
    left_part = arr[:L-1]
    middle_part = arr[L-1:R]
    right_part = arr[R:]
    reversed_middle = middle_part[::-1]
    result = left_part + reversed_middle + right_part
    print(' '.join(map(str, result)))

if __name__ == "__main__":
    main()