N, K = map(int, input().split())
A = list(map(int, input().split()))

# Sum of all integers from 1 to K
total_sum = K * (K + 1) // 2

# Find unique numbers in A that are between 1 and K
present_in_range = set()
for num in A:
    if 1 <= num <= K:
        present_in_range.add(num)

# Sum of numbers present in A and within range [1, K]
present_sum = sum(present_in_range)

# Answer is total sum minus present sum
answer = total_sum - present_sum
print(answer)