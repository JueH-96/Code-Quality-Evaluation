# YOUR CODE HERE
def count_vertices(N, X, K):
    def get_parent(x):
        return x // 2
    
    def get_children(x):
        return [2 * x, 2 * x + 1]
    
    def count_at_distance(x, k, depth=0):
        if k == 0:
            return 1
        if x > N:
            return 0
        
        count = 0
        for child in get_children(x):
            if child <= N:
                count += count_at_distance(child, k - 1, depth + 1)
        
        if depth > 0:
            parent = get_parent(x)
            if parent > 0:
                count += count_at_distance(parent, k - 1, depth + 1)
        
        return count
    
    return count_at_distance(X, K)

import sys
input = sys.stdin.read
data = input().split()

T = int(data[0])
index = 1
results = []

for _ in range(T):
    N = int(data[index])
    X = int(data[index + 1])
    K = int(data[index + 2])
    index += 3
    results.append(count_vertices(N, X, K))

for result in results:
    print(result)