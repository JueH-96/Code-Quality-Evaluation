import sys

def main():
    data = sys.stdin.buffer.read().split()
    if not data:
        return
    it = iter(data)
    n = int(next(it))
    A = list(map(int, (next(it) for _ in range(n))))

    # prefix_distinct[i] = number of distinct values in A[0..i]
    seen = [0] * (n + 1)
    prefix = [0] * n
    cnt = 0
    for i, x in enumerate(A):
        if seen[x] == 0:
            seen[x] = 1
            cnt += 1
        prefix[i] = cnt

    # suffix_distinct[i] = number of distinct values in A[i..n-1]
    seen2 = [0] * (n + 1)
    suffix = [0] * n
    cnt2 = 0
    for i in range(n - 1, -1, -1):
        x = A[i]
        if seen2[x] == 0:
            seen2[x] = 1
            cnt2 += 1
        suffix[i] = cnt2

    # maximize prefix[i] + suffix[i+1] for i = 0..n-2
    ans = 0
    for i in range(n - 1):
        total = prefix[i] + suffix[i + 1]
        if total > ans:
            ans = total

    print(ans)

if __name__ == "__main__":
    main()