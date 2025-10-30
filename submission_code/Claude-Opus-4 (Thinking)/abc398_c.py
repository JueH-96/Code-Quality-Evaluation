# YOUR CODE HERE
n = int(input())
a = list(map(int, input().split()))

# Count frequency of each integer
freq = {}
for num in a:
    if num in freq:
        freq[num] += 1
    else:
        freq[num] = 1

# Find the maximum unique integer and its index
max_unique = -1
max_index = -1

for i in range(n):
    if freq[a[i]] == 1:
        if a[i] > max_unique:
            max_unique = a[i]
            max_index = i + 1

if max_index == -1:
    print(-1)
else:
    print(max_index)