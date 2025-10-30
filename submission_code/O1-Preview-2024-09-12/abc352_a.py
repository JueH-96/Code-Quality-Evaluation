# YOUR CODE HERE
N, X, Y, Z = map(int, input().split())

if X < Y:
    route = range(X, Y+1)  # Inbound train
else:
    route = range(X, Y-1, -1)  # Outbound train

if Z in route:
    print('Yes')
else:
    print('No')