def solve():
    n, k = map(int, input().split())
    s = input()
    
    black_indices = [i for i, char in enumerate(s) if char == 'B']
    
    if not black_indices:
        print(0)
        return
    
    operations = 0
    i = 0
    while i < len(black_indices):
        operations += 1
        i = black_indices[i] + k
        while i < len(black_indices) and black_indices[i] < black_indices[i-1] + k:
            i += 1
    
    print(operations)

t = int(input())
for _ in range(t):
    solve()