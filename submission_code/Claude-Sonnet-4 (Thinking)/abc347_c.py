N, A, B = map(int, input().split())
D = list(map(int, input().split()))

W = A + B

if A >= W:
    print("Yes")
else:
    # Collect all critical points
    critical = set()
    for d in D:
        start = (-d) % W
        critical.add(start)
        critical.add((start + A) % W)
    
    # Check each critical point
    for Y in critical:
        if all((Y + d) % W <= A - 1 for d in D):
            print("Yes")
            exit()
    
    print("No")