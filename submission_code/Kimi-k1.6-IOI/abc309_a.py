A, B = map(int, input().split())
print("Yes" if B == A + 1 and A not in {3, 6, 9} else "No")