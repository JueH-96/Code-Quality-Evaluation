from collections import Counter

def is_possible(s, k):
    n = len(s)
    counter = Counter(s)
    odd_count = sum(1 for key in counter if counter[key] % 2)
    return odd_count <= k

t = int(input().strip())
for _ in range(t):
    n, k = map(int, input().strip().split())
    s = input().strip()
    print("YES" if is_possible(s, k) else "NO")