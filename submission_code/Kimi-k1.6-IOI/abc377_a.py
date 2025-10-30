S = input().strip()
sorted_S = sorted(S)
print("Yes" if ''.join(sorted_S) == 'ABC' else "No")