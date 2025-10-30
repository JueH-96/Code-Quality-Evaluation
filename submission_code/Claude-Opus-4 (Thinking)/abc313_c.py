# YOUR CODE HERE
N = int(input())
A = list(map(int, input().split()))

S = sum(A)
q = S // N  # floor(S/N)
r = S % N   # remainder

# Sort the array
A.sort()

# The r largest elements should have target q+1
# The N-r smallest elements should have target q
cost = 0
for i in range(N):
    if i < N - r:
        # This element should have target q
        cost += max(0, A[i] - q)
    else:
        # This element should have target q+1
        cost += max(0, A[i] - (q + 1))

print(cost)