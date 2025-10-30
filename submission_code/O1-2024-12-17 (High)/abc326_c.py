def main():
    import sys
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    M = int(data[1])
    A = list(map(int, data[2:]))

    A.sort()

    left = 0
    answer = 1
    for right in range(N):
        while A[right] >= A[left] + M:
            left += 1
        answer = max(answer, right - left + 1)

    print(answer)

if __name__ == "__main__":
    main()