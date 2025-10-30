N, X, Y, Z = map(int, input().split())

if (X < Y and (X < Z < Y)) or (X > Y and (X > Z > Y)):
    print('Yes')
else:
    print('No')