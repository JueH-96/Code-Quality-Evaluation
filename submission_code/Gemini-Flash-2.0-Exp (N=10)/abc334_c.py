def solve():
    n, k = map(int, input().split())
    a = list(map(int, input().split()))

    counts = [2] * (n + 1)
    for x in a:
        counts[x] -= 1

    socks = []
    for i in range(1, n + 1):
        socks.extend([i] * counts[i])
    
    num_pairs = len(socks) // 2
    
    socks.sort()
    
    total_weirdness = 0
    for i in range(num_pairs):
        total_weirdness += abs(socks[2*i] - socks[2*i+1])
        
    print(total_weirdness)

solve()