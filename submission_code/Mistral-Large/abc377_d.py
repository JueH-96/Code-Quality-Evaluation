import sys
input = sys.stdin.read

def solve():
    data = input().split()
    index = 0

    N = int(data[index])
    M = int(data[index + 1])
    index += 2

    intervals = []
    for _ in range(N):
        L = int(data[index])
        R = int(data[index + 1])
        index += 2
        intervals.append((L, R))

    intervals.sort(key=lambda x: (x[1], x[0]))

    count = [0] * (M + 2)
    current_end = 0

    for i in range(1, M + 1):
        count[i] = count[i - 1]
        while current_end < N and intervals[current_end][1] < i:
            current_end += 1
        if current_end < N and intervals[current_end][0] <= i <= intervals[current_end][1]:
            count[i] += 1

    total_pairs = 0
    for r in range(1, M + 1):
        l = r
        while l > 0 and count[l] == count[r]:
            l -= 1
        total_pairs += r - l

    print(total_pairs)

if __name__ == "__main__":
    solve()