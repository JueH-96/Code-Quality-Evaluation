def main():
    N = int(input())
    heights = list(map(int, input().split()))

    counts = [0] * N
    for i in range(N):
        for j in range(i + 1, N):
            if heights[j] > heights[i]:
                counts[i] += 1
            else:
                break

    print(*counts)

if __name__ == "__main__":
    main()