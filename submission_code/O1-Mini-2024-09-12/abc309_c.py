import sys

def main():
    import sys
    import bisect

    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    K = int(data[1])
    a_b = list(map(int, data[2:]))
    a = []
    b = []
    for i in range(N):
        a.append(a_b[2*i])
        b.append(a_b[2*i+1])
    combined = sorted(zip(a, b))
    sorted_a = [x[0] for x in combined]
    sorted_b = [x[1] for x in combined]
    suffix_sum = [0]*(N+1)
    for i in range(N-1, -1, -1):
        suffix_sum[i] = suffix_sum[i+1] + sorted_b[i]
    left = 1
    right = max(sorted_a) + 1
    answer = right
    while left <= right:
        mid = (left + right) // 2
        idx = bisect.bisect_left(sorted_a, mid)
        total = suffix_sum[idx]
        if total <= K:
            answer = mid
            right = mid -1
        else:
            left = mid +1
    print(answer)

if __name__ == "__main__":
    main()