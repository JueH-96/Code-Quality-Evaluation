A = list(map(int, input().split()))
sorted_A = [1, 2, 3, 4, 5]
found = False

for i in range(4):
    new_A = A.copy()
    new_A[i], new_A[i+1] = new_A[i+1], new_A[i]
    if new_A == sorted_A:
        found = True
        break

print("Yes" if found else "No")