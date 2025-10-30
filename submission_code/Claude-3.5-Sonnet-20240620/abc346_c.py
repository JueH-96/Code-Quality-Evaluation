# YOUR CODE HERE
N, K = map(int, input().split())
A = set(map(int, input().split()))

total_sum = K * (K + 1) // 2  # Sum of integers from 1 to K
sum_in_A = sum(a for a in A if a <= K)

result = total_sum - sum_in_A
print(result)