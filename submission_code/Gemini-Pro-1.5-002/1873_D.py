# YOUR CODE HERE
def solve():
    n, k = map(int, input().split())
    s = input()
    
    black_indices = [i for i, char in enumerate(s) if char == 'B']
    
    if not black_indices:
        print(0)
        return
    
    count = 0
    covered = [False] * len(black_indices)
    
    for i in range(len(black_indices)):
        if not covered[i]:
            count += 1
            for j in range(i, len(black_indices)):
                if black_indices[j] - black_indices[i] < k:
                    covered[j] = True
                else:
                    break
    
    print(count)

t = int(input())
for _ in range(t):
    solve()