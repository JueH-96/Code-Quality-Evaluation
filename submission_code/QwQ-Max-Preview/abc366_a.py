N, T, A = map(int, input().split())
remaining = N - T - A

if remaining == 0:
    print("Yes")
else:
    if T > A:
        if A + remaining < T:
            print("Yes")
        else:
            print("No")
    elif A > T:
        if T + remaining < A:
            print("Yes")
        else:
            print("No")
    else:
        print("No")