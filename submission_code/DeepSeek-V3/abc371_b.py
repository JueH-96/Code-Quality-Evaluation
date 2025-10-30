# YOUR CODE HERE
# Read input values
N, M = map(int, input().split())
A = []
B = []
for _ in range(M):
    a, b = input().split()
    A.append(int(a))
    B.append(b)

# Initialize a dictionary to keep track of the first male in each family
first_male = {}

# Iterate through each baby
for i in range(M):
    family = A[i]
    gender = B[i]
    if gender == 'M':
        if family not in first_male:
            first_male[family] = i
            print("Yes")
        else:
            print("No")
    else:
        print("No")