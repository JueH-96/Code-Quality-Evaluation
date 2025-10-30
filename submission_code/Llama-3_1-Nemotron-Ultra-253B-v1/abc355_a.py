a, b = map(int, input().split())
possible = [x for x in [1, 2, 3] if x != a and x != b]
print(possible[0] if len(possible) == 1 else -1)