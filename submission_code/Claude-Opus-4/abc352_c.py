# YOUR CODE HERE
n = int(input())
giants = []
for i in range(n):
    a, b = map(int, input().split())
    giants.append((a, b))

# The height of the topmost giant is:
# sum(A_i for all i except the top one) + B_top
# = sum(all A_i) - A_top + B_top
# = sum(all A_i) + (B_top - A_top)

# To maximize this, we want to maximize (B_top - A_top)
# So the top giant should be the one with max(B_i - A_i)

total_a = sum(a for a, b in giants)
max_diff = max(b - a for a, b in giants)

print(total_a + max_diff)