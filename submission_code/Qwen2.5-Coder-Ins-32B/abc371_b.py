# YOUR CODE HERE
import sys

input = sys.stdin.read().split()
N = int(input[0])
M = int(input[1])

# Track the first male child for each family
first_male = [False] * (N + 1)
results = []

index = 2
for _ in range(M):
    family = int(input[index])
    gender = input[index + 1]
    index += 2
    
    if gender == 'M' and not first_male[family]:
        results.append('Yes')
        first_male[family] = True
    else:
        results.append('No')

print('
'.join(results))