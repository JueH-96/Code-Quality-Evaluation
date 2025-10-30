# YOUR CODE HERE
def solve():
    n = int(input())
    a = list(map(int, input().split()))
    
    f = {}
    for i in range(1, n + 1):
        indices = []
        for j in range(3 * n):
            if a[j] == i:
                indices.append(j + 1)
        f[i] = indices[1]
        
    sorted_keys = sorted(f, key=f.get)
    
    print(*sorted_keys)

solve()