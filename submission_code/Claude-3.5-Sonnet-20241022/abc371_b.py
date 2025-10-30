N, M = map(int, input().split())

# Keep track of first male child in each family
first_male = [False] * (N+1)

# Process each baby
for i in range(M):
    family, gender = input().split()
    family = int(family)
    
    # If male baby and no previous male in family
    if gender == 'M' and not first_male[family]:
        first_male[family] = True
        print("Yes")
    else:
        print("No")