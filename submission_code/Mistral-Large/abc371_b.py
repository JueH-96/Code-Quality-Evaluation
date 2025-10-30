import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
M = int(data[1])

# Initialize a list to keep track of whether a family has had a male child
family_has_male = [False] * N

results = []

for i in range(M):
    A_i = int(data[2 * i + 2]) - 1  # Convert to 0-based index
    B_i = data[2 * i + 3]

    if B_i == 'M' and not family_has_male[A_i]:
        family_has_male[A_i] = True
        results.append("Yes")
    else:
        results.append("No")

# Print the results
for result in results:
    print(result)