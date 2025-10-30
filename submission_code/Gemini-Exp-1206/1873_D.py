def solve():
    n, k = map(int, input().split())
    s = input()
    
    black_indices = [i for i, char in enumerate(s) if char == 'B']
    
    if not black_indices:
        print(0)
        return
    
    operations = 0
    last_covered = -1
    
    for i in black_indices:
        if i > last_covered:
            operations += 1
            last_covered = i + k - 1
    
    print(operations)

t = int(input())
for _ in range(t):
    solve()