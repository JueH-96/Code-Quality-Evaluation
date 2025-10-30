N, K, Q = map(int, input().split())
A = [0] * (N + 1)  # 1-indexed

for _ in range(Q):
    X, Y = map(int, input().split())
    A[X] = Y
    
    # Get all elements and sort in descending order
    elements = []
    for i in range(1, N + 1):
        elements.append(A[i])
    
    elements.sort(reverse=True)
    
    # Sum the first K elements
    result = sum(elements[:K])
    print(result)