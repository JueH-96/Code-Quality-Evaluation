def solve():
    n, k = map(int, input().split())
    s = input()
    char_counts = {}
    for char in s:
        char_counts[char] = char_counts.get(char, 0) + 1
    
    initial_odd_count = 0
    for char in char_counts:
        if char_counts[char] % 2 != 0:
            initial_odd_count += 1
            
    if initial_odd_count <= k + 1:
        print("YES")
    else:
        print("NO")

t = int(input())
for _ in range(t):
    solve()