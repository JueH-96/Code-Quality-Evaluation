A = list(map(int, input().split()))
target = [1, 2, 3, 4, 5]

for i in range(4):
    new_A = A.copy()
    new_A[i], new_A[i+1] = new_A[i+1], new_A[i]
    if new_A == target:
        print("Yes")
        exit()
print("No")