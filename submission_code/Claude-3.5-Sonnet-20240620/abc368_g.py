# YOUR CODE HERE
def max_value(A, B, l, r):
    v = 0
    for i in range(l, r + 1):
        v = max(v + A[i], v * B[i])
    return v

N = int(input())
A = [0] + list(map(int, input().split()))
B = [0] + list(map(int, input().split()))
Q = int(input())

for _ in range(Q):
    query = list(map(int, input().split()))
    
    if query[0] == 1:
        _, i, x = query
        A[i] = x
    elif query[0] == 2:
        _, i, x = query
        B[i] = x
    else:  # query[0] == 3
        _, l, r = query
        print(max_value(A, B, l, r))