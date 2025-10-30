from math import floor

def solve_test_case():
    N, X, K = map(int, input().split())
    count = 0
    
    # Traverse the tree from the root (1) to the target vertex (X)
    # and count the number of vertices at distance K from X
    current = X
    for _ in range(K):
        current = floor(current / 2)
    
    # Count the number of vertices at distance K from X
    while current >= 1:
        if abs(current - X) == K:
            count += 1
        current = floor(current / 2)
    
    return count

T = int(input())
for _ in range(T):
    print(solve_test_case())