def solve():
    n, k = map(int, input().split())
    s = input()
    
    counts = {}
    for char in s:
        counts[char] = counts.get(char, 0) + 1
    
    odd_count = 0
    for char in counts:
        if counts[char] % 2 != 0:
            odd_count += 1
    
    if odd_count - 1 <= k:
        print("YES")
    else:
        print("NO")

t = int(input())
for _ in range(t):
    solve()