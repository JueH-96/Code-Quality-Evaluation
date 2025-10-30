import sys
data = sys.stdin.read().split()
index = 0
N = int(data[index])
index += 1
A = list(map(int, data[index:index + N]))

# Compute the cumulative sum and find the minimum prefix sum
current_sum = 0
min_sum = float('inf')
for a in A:
    current_sum += a
    if current_sum < min_sum:
        min_sum = current_sum

# Calculate the minimum initial passengers X
min_X = max(0, -min_sum)

# Calculate the answer: min_X + total sum
answer = min_X + current_sum

# Output the answer
print(answer)