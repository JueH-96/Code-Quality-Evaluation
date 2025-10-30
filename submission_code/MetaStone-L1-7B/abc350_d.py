def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    M = int(data[1])
    degrees = [0] * (N + 1)
    index = 2
    for _ in range(M):
        A = int(data[index])
        B = int(data[index + 1])
        degrees[A] += 1
        degrees[B] += 1
        index += 2

    total = 0
    for d in degrees:
        if d >= 2:
            total += d * (d - 1) // 2

    answer = total - M
    print(max(answer, 0))

if __name__ == '__main__':
    main()