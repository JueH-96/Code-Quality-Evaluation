# Read the input
N, M = map(int, input().split())
families = [0] * N  # Keep track of the eldest son in each family
for i in range(M):
    A, B = map(int, input().split())
    if B == 'M' and families[A-1] == 0:
        families[A-1] = 1  # Mark the eldest son in this family
    if families[A-1] == 1:
        print("Yes")
    else:
        print("No")