A, B = map(int, input().split())
possible = [x for x in [1, 2, 3] if x != A and x != B]
if len(possible) == 1:
    print(possible[0])
else:
    print(-1)