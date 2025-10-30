# YOUR CODE HERE
n = int(input())
a = list(map(int, input().split()))

# Count occurrences of each integer
count = {}
for num in a:
    count[num] = count.get(num, 0) + 1

# Find people with unique integers
max_value = -1
max_person = -1

for i in range(n):
    if count[a[i]] == 1:
        if a[i] > max_value:
            max_value = a[i]
            max_person = i + 1  # 1-indexed

print(max_person)