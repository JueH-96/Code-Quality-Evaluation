# Read the input
N = int(input())
A = list(map(int, input().split()))

# Create a dictionary to store the count of each integer
count = {}
for a in A:
    count[a] = count.get(a, 0) + 1

# Find the person with the largest integer that satisfies the condition
max_person = -1
max_value = -1
for i, a in enumerate(A):
    if count[a] == 1:
        if a > max_value:
            max_person = i + 1
            max_value = a

# Print the answer
if max_person == -1:
    print(-1)
else:
    print(max_person)