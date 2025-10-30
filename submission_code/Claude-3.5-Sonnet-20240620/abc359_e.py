# YOUR CODE HERE
N = int(input())
H = list(map(int, input().split()))

result = [0] * N
water = 0
for i in range(N):
    water += 1
    if water > H[i]:
        result[i] = water
        water = H[i]
    else:
        result[i] = H[i] + 1

cumulative_sum = 0
for i in range(N):
    cumulative_sum += result[i]
    result[i] = cumulative_sum

print(*result)