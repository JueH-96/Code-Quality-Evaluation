# YOUR CODE HERE
n = int(input())
p = list(map(int, input().split()))

total_cost = 0

# For each position i (1-indexed from 1 to n-1)
for i in range(1, n):
    count = 0
    # Count elements at positions > i that have values <= i
    for j in range(i, n):  # positions i+1 to n in 1-indexed (i to n-1 in 0-indexed)
        if p[j] <= i:  # if element value <= i, it needs to move left past position i
            count += 1
    total_cost += count * i

print(total_cost)