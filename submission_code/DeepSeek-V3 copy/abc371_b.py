# Read input values
N, M = map(int, input().split())

# Initialize a dictionary to keep track of whether a family has had a male child
family_has_male = {i: False for i in range(1, N+1)}

# Process each baby
for _ in range(M):
    A, B = input().split()
    A = int(A)
    if B == 'M':
        if not family_has_male[A]:
            print("Yes")
            family_has_male[A] = True
        else:
            print("No")
    else:
        print("No")