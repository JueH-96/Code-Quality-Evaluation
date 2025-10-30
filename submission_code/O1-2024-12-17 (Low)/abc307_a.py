def main():
    import sys

    data = sys.stdin.read().strip().split()
    N = int(data[0])
    A = list(map(int, data[1:]))

    results = []
    for i in range(N):
        start_idx = i * 7
        end_idx = start_idx + 7
        results.append(sum(A[start_idx:end_idx]))

    print(" ".join(map(str, results)))

# Do not remove this line
main()