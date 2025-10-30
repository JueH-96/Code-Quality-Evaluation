# YOUR CODE HERE
def min_weirdness(N, K, A):
    colors = set(range(1, N+1))
    lost_colors = set(A)
    remaining_colors = sorted(colors - lost_colors)
    
    total_socks = 2*N - K
    pairs = total_socks // 2
    
    weirdness = 0
    for i in range(0, len(remaining_colors) - 1, 2):
        if pairs == 0:
            break
        weirdness += abs(remaining_colors[i] - remaining_colors[i+1])
        pairs -= 1
    
    return weirdness

N, K = map(int, input().split())
A = list(map(int, input().split()))

print(min_weirdness(N, K, A))