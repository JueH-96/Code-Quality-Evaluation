def solve():
    n = int(input())
    a = list(map(int, input().split()))
    
    max_val = 10**6
    freq = [0] * (max_val + 1)
    for x in a:
        freq[x] += 1
        
    p = [0] * (max_val + 2)
    p[max_val + 1] = 0
    
    for k in range(max_val, 0, -1):
        p[k] = k * freq[k] + p[k+1]
        
    results = []
    for i in range(n):
        results.append(p[a[i] + 1])
        
    print(*(results))

if __name__ == '__main__':
    solve()