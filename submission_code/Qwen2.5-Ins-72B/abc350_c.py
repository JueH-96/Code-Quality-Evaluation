# YOUR CODE HERE
def solve():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    A = list(map(int, data[1:]))
    
    target = list(range(1, N + 1))
    operations = []
    
    for i in range(N):
        if A[i] != target[i]:
            j = A.index(target[i])
            A[i], A[j] = A[j], A[i]
            operations.append((i + 1, j + 1))
    
    print(len(operations))
    for op in operations:
        print(op[0], op[1])

solve()