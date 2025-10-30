# YOUR CODE HERE
a, b = map(int, input().split())

suspects = [1, 2, 3]
suspects.remove(a)
suspects.remove(b)

if len(suspects) == 1:
    print(suspects[0])
elif a == b and len(suspects) ==2:
    print(-1)
elif a != b and len(suspects) == 2:
    print(-1)
else:
    print(-1)