A, B = map(int, input().split())

suspects = [1, 2, 3]
suspects.remove(A)
suspects.remove(B)

if len(suspects) == 1:
    print(suspects[0])
else:
    print(-1)