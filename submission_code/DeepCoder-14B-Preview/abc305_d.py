import bisect

def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0

    N = int(input[ptr])
    ptr += 1

    A = list(map(int, input[ptr:ptr + N]))
    ptr += N

    Q = int(input[ptr])
    ptr += 1

    queries = []
    for _ in range(Q):
        l = int(input[ptr])
        r = int(input[ptr + 1])
        queries.append((l, r))
        ptr += 2

    times = A
    prefix = [0] * len(times)

    for i in range(1, len(times)):
        duration = times[i] - times[i - 1]
        if (i - 1) % 2 == 1:
            prefix[i] = prefix[i - 1] + duration
        else:
            prefix[i] = prefix[i - 1]

    def compute_sleep(t):
        if t <= 0:
            return 0
        i = bisect.bisect_right(times, t) - 1
        if i < 0:
            return 0
        sleep_time = prefix[i]
        if i % 2 == 1:
            sleep_time += (t - times[i])
        return sleep_time

    results = []
    for l, r in queries:
        ans = compute_sleep(r) - compute_sleep(l)
        results.append(ans)

    for res in results:
        print(res)

if __name__ == '__main__':
    main()