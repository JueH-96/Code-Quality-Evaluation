A, B = map(int, input().split())

if A == B:
    print(1)
elif (A + B) % 2 == 0:
    print(3)
else:
    print(2)