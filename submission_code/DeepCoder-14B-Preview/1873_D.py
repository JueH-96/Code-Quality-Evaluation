import bisect

def min_operations():
    t = int(input())
    for _ in range(t):
        n, k = map(int, input().split())
        s = input().strip()
        bs = [i for i, c in enumerate(s) if c == 'B']
        if not bs:
            print(0)
            continue
        count = 0
        i = 0
        while i < len(bs):
            current = bs[i]
            end = current + k - 1
            # Find the rightmost B that is <= end
            j = bisect.bisect_right(bs, end) - 1
            count += 1
            i = j + 1
        print(count)

min_operations()