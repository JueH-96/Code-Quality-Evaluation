# YOUR CODE HERE
t = int(input())
for _ in range(t):
    n, k = map(int, input().split())
    s = input()
    from collections import Counter
    counts = Counter(s)
    O = sum(1 for c in counts if counts[c] % 2 == 1)
    if k >= O - 1:
        print("YES")
    else:
        print("NO")