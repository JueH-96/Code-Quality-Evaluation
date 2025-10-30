# YOUR CODE HERE
n = int(input())
a = list(map(int, input().split()))

# Count frequency of each integer
freq = {}
for num in a:
    freq[num] = freq.get(num, 0) + 1

# Find people with unique integers
unique_people = []
for i in range(n):
    if freq[a[i]] == 1:
        unique_people.append((a[i], i + 1))  # (value, 1-indexed label)

if not unique_people:
    print(-1)
else:
    # Find person with maximum integer among unique people
    max_value = max(unique_people)[0]
    for value, label in unique_people:
        if value == max_value:
            print(label)
            break