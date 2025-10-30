# YOUR CODE HERE
N = int(input())
A = list(map(int, input().split()))

# Calculate the sum without modulo
total_sum = (N - 1) * sum(A)

# Count pairs with sum >= 10^8
sorted_A = sorted(A)
left = 0
right = N - 1
count = 0
while left < right:
    if sorted_A[left] + sorted_A[right] >= 10**8:
        count += right - left
        right -= 1
    else:
        left += 1

# The answer
answer = total_sum - count * 10**8
print(answer)