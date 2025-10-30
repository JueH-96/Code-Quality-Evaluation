n = int(input())
a = list(map(int, input().split()))

# Count frequency of each integer
freq = {}
for value in a:
    freq[value] = freq.get(value, 0) + 1

# Find the person with the unique value that is maximum
max_value = -1
result = -1

for i in range(n):
    if freq[a[i]] == 1 and a[i] > max_value:
        max_value = a[i]
        result = i + 1

print(result)