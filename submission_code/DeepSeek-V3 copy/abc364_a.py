# Read the number of dishes
N = int(input())
# Read the list of dishes
dishes = [input().strip() for _ in range(N)]
# Initialize a flag to check if two consecutive sweet dishes are found
can_eat_all = True
# Iterate through the dishes
for i in range(1, N):
    if dishes[i] == 'sweet' and dishes[i-1] == 'sweet':
        can_eat_all = False
        break
# Print the result
if can_eat_all:
    print("Yes")
else:
    print("No")