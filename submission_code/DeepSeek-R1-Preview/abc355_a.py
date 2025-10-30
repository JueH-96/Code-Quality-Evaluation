A, B = map(int, input().split())
culprits = [x for x in [1, 2, 3] if x != A and x != B]
print(culprits[0] if len(culprits) == 1 else -1)