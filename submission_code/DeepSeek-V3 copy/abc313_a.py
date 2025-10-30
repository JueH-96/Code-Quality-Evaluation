# Read the number of people
N = int(input())
# Read the list of programming abilities
P = list(map(int, input().split()))
# Get the programming ability of person 1
P1 = P[0]
# Get the maximum programming ability among the others
max_other = max(P[1:]) if N > 1 else 0
# Calculate the required additional points
if P1 > max_other:
    print(0)
else:
    print(max_other - P1 + 1)