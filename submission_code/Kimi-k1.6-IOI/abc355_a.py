A, B = map(int, input().split())
possible = [x for x in [1, 2, 3] if x != A and x != B]
print(possible[0] if len(possible) == 1 else -1)