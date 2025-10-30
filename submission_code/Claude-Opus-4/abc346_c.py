# YOUR CODE HERE
N, K = map(int, input().split())
A = list(map(int, input().split()))

# Total sum from 1 to K
total_sum = K * (K + 1) // 2

# Find unique elements in A that are between 1 and K
unique_in_range = set()
for num in A:
    if 1 <= num <= K:
        unique_in_range.add(num)

# Sum of elements that appear in A and are in range [1, K]
sum_in_A = sum(unique_in_range)

# Answer is the difference
answer = total_sum - sum_in_A
print(answer)