# YOUR CODE HERE
import sys
from collections import defaultdict

def min_good_index_diff(N, K, P):
    if K == 1:
        return 0
    
    index_map = defaultdict(list)
    for i, p in enumerate(P):
        index_map[p].append(i)
    
    min_diff = float('inf')
    
    for start in range(1, N - K + 2):
        indices = []
        for i in range(K):
            if start + i in index_map:
                indices.append(index_map[start + i][0])
            else:
                break
        if len(indices) == K:
            min_diff = min(min_diff, max(indices) - min(indices))
    
    return min_diff

input = sys.stdin.read
data = input().split()
N = int(data[0])
K = int(data[1])
P = list(map(int, data[2:]))

print(min_good_index_diff(N, K, P))