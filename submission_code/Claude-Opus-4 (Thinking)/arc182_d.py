# YOUR CODE HERE
n, m = map(int, input().split())
a = list(map(int, input().split()))
b = list(map(int, input().split()))

if m == 2:
    if a == b:
        print(0)
    else:
        print(-1)
else:
    total = 0
    for i in range(n):
        # Calculate clockwise and counterclockwise distances
        dist_clockwise = (b[i] - a[i]) % m
        dist_counter = (a[i] - b[i]) % m
        # Take the minimum distance
        total += min(dist_clockwise, dist_counter)
    
    print(total)