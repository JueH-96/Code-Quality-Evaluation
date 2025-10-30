def main():
    import sys
    input = sys.stdin.read().split()
    n = int(input[0])
    data = input[1:]
    sum_a = 0
    max_diff = 0
    for i in range(n):
        a = int(data[2*i])
        b = int(data[2*i + 1])
        sum_a += a
        diff = b - a
        if diff > max_diff:
            max_diff = diff
    print(sum_a + max_diff)

if __name__ == "__main__":
    main()