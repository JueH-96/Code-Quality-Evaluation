# YOUR CODE HERE
n = int(input())
a = list(map(int, input().split()))

# Find the second largest value
# Method: find max, then find max excluding the first max
max_val = max(a)
second_max = max(x for x in a if x != max_val)

# Find the index of second_max (1-based)
for i in range(n):
    if a[i] == second_max:
        print(i + 1)
        break