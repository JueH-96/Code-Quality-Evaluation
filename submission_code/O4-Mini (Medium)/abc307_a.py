def main():
    import sys
    data = sys.stdin.read().split()
    N = int(data[0])
    A = list(map(int, data[1:]))
    # There should be exactly 7*N step counts
    result = []
    for i in range(N):
        week_sum = sum(A[7*i : 7*i + 7])
        result.append(str(week_sum))
    print(" ".join(result))

main()