N, Q = map(int, input().split())
for _ in range(Q):
    K = int(input())
    if N == 2:
        if K == 0 or K == 4:
            print("Yes")
        else:
            print("No")
    elif N == 3:
        if K == 0 or K == 9:
            print("Yes")
        else:
            print("No")
    else:
        if K == 0 or K == N**2:
            print("Yes")
        else:
            print("No")