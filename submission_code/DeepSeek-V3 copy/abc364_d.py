import sys
import bisect

def main():
    input = sys.stdin.read
    data = input().split()
    idx = 0
    N = int(data[idx])
    Q = int(data[idx+1])
    idx += 2
    a = list(map(int, data[idx:idx+N]))
    idx += N
    queries = []
    for _ in range(Q):
        b = int(data[idx])
        k = int(data[idx+1])
        queries.append((b, k))
        idx += 2
    a.sort()
    for b, k in queries:
        left = bisect.bisect_left(a, b)
        right = left
        distances = []
        for i in range(max(0, left - 100), min(N, left + 100)):
            distances.append(abs(a[i] - b))
        distances.sort()
        print(distances[k-1])

if __name__ == "__main__":
    main()