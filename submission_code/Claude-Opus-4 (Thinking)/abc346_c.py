# YOUR CODE HERE
N, K = map(int, input().split())
A = list(map(int, input().split()))

# Get unique elements in A that are between 1 and K
in_range_elements = set()
for a in A:
    if 1 <= a <= K:
        in_range_elements.add(a)

# Sum of all integers from 1 to K
total_sum = K * (K + 1) // 2

# Sum of elements in A that are between 1 and K
sum_in_A = sum(in_range_elements)

# Answer is the difference
answer = total_sum - sum_in_A
print(answer)