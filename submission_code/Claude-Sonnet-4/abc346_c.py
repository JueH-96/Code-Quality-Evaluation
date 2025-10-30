# YOUR CODE HERE
n, k = map(int, input().split())
a = list(map(int, input().split()))

# Sum of integers from 1 to K
total_sum = k * (k + 1) // 2

# Find unique elements in A that are between 1 and K
present_in_range = set()
for num in a:
    if 1 <= num <= k:
        present_in_range.add(num)

# Sum of elements that are present in the range [1, K]
present_sum = sum(present_in_range)

# Answer is total sum minus sum of present elements
answer = total_sum - present_sum

print(answer)