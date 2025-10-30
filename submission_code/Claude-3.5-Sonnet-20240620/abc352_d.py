# YOUR CODE HERE
def find_min_range(N, K, P):
    positions = [0] * (N + 1)
    for i, p in enumerate(P):
        positions[p] = i + 1
    
    min_range = N
    for start in range(1, N - K + 2):
        end = start + K - 1
        max_pos = max(positions[start:end+1])
        min_pos = min(positions[start:end+1])
        min_range = min(min_range, max_pos - min_pos)
    
    return min_range

N, K = map(int, input().split())
P = list(map(int, input().split()))

result = find_min_range(N, K, P)
print(result)