def solve():
    n = int(input())
    P = list(map(int, input().split()))
    Q = list(map(int, input().split()))
    
    # Create inverse permutations
    inv_P = [0] * (n + 1)
    inv_Q = [0] * (n + 1)
    
    for i in range(n):
        inv_P[P[i]] = i + 1
        inv_Q[Q[i]] = i + 1
    
    # Construct the matrix
    A = []
    for i in range(1, n + 1):
        row = []
        for j in range(1, n + 1):
            if inv_P[i] >= inv_Q[j]:
                row.append('1')
            else:
                row.append('0')
        A.append(''.join(row))
    
    for row in A:
        print(row)

solve()