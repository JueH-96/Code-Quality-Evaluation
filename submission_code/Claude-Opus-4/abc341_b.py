# YOUR CODE HERE
N = int(input())
A = list(map(int, input().split()))

# Read exchange rates
exchanges = []
for i in range(N-1):
    s, t = map(int, input().split())
    exchanges.append((s, t))

# Work from country 1 to N-1
for i in range(N-1):
    s_i, t_i = exchanges[i]
    # How many times can we exchange?
    times = A[i] // s_i
    if times > 0:
        # Perform the exchange
        A[i] -= times * s_i
        A[i+1] += times * t_i

print(A[N-1])