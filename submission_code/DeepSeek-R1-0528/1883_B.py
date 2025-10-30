t = int(input().strip())
for _ in range(t):
    n, k = map(int, input().split())
    s = input().strip()
    m = n - k
    from collections import Counter
    freq = Counter(s)
    odd_count = sum(1 for count in freq.values() if count % 2 == 1)
    base = n - odd_count

    if m % 2 == 0:
        if m <= base:
            print("YES")
        else:
            print("NO")
    else:
        if odd_count > 0:
            if m <= base + 1:
                print("YES")
            else:
                print("NO")
        else:
            print("YES")