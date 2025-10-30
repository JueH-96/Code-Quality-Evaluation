def main():
    import sys
    input = sys.stdin.read().split()
    n = int(input[0])
    a = list(map(int, input[1:n+1]))
    last_occurrence = {}
    b = []
    for i in range(n):
        num = a[i]
        if num in last_occurrence:
            b.append(last_occurrence[num])
        else:
            b.append(-1)
        last_occurrence[num] = i + 1
    print(' '.join(map(str, b)))

if __name__ == "__main__":
    main()