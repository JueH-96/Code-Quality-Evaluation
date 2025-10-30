# Read the input values
L, R = map(int, input().split())

# Check the sum of L and R
sum_val = L + R

if sum_val != 1:
    print("Invalid")
else:
    if L == 1:
        print("Yes")
    else:
        print("No")