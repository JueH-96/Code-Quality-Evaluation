t = int(input())
for _ in range(t):
    n, k = map(int, input().split())
    s = input().strip()
    freq = [0] * 26
    for c in s:
        freq[ord(c) - ord('a')] += 1
    O_initial = sum(f % 2 for f in freq)
    m = n - k
    required_O = 0 if (m % 2 == 0) else 1
    delta = required_O - O_initial
    if (delta % 2 == k % 2) and (abs(delta) <= k):
        print("YES")
    else:
        print("NO")