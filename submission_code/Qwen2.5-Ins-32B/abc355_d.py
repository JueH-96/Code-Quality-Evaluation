import sys

def count_intersecting_intervals(N, intervals):
    intervals.sort(key=lambda x: x[0])
    count = 0
    for i in range(N):
        l_i, r_i = intervals[i]
        for j in range(i + 1, N):
            l_j, r_j = intervals[j]
            if r_i < l_j:
                break
            if l_i <= l_j <= r_i or l_i <= r_j <= r_i:
                count += 1
    return count

if __name__ == "__main__":
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    intervals = [(int(data[i]), int(data[i + 1])) for i in range(1, 2 * N, 2)]
    print(count_intersecting_intervals(N, intervals))