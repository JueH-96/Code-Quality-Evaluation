def main():
    import sys
    input_data = sys.stdin.read().split()
    n = int(input_data[0])
    a = list(map(int, input_data[1:]))
    result = []
    for i in range(n - 1):
        result.append(a[i] * a[i + 1])
    print(" ".join(map(str, result)))

if __name__ == '__main__':
    main()