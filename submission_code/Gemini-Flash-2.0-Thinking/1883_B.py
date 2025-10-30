def solve():
    n, k = map(int, input().split())
    s = input()

    remaining_len = n - k

    from collections import Counter
    counts = Counter(s)
    odd_counts_orig = 0
    for count in counts.values():
        if count % 2 != 0:
            odd_counts_orig += 1

    if remaining_len % 2 == 0:
        # Remaining string needs 0 odd frequency characters
        if k >= odd_counts_orig and (k - odd_counts_orig) % 2 == 0:
            print("YES")
        else:
            print("NO")
    else:
        # Remaining string needs 1 odd frequency character
        if k + 1 >= odd_counts_orig and (k + 1 - odd_counts_orig) % 2 == 0:
            print("YES")
        else:
            print("NO")

t = int(input())
for _ in range(t):
    solve()