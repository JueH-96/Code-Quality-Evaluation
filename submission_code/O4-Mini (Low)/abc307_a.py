def main():
    import sys
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    A = list(map(int, data[1:]))

    # Compute weekly sums
    weekly_sums = []
    for week in range(N):
        start = week * 7
        end = start + 7
        weekly_sums.append(str(sum(A[start:end])))

    # Output results
    print(" ".join(weekly_sums))

if __name__ == "__main__":
    main()