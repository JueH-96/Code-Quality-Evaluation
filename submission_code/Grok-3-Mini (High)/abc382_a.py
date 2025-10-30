import sys
data = sys.stdin.read().split()
index = 0
N = int(data[index])
D = int(data[index + 1])
index += 2
S = data[index]

# Count the number of cookies initially
C = S.count('@')

# Calculate the number of empty boxes after D days
answer = N - C + D

# Output the answer
print(answer)