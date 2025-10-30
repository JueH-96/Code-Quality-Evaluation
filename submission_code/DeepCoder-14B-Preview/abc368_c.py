# Precompute Grundy numbers up to 100,000
max_n = 10**5
G = [0] * (max_n + 1)
G[1] = 0  # Base case

for n in range(2, max_n + 1):
    divisors = set()
    if n > 1:
        divisors.add(1)
    for d in range(2, int(n**0.5) + 1):
        if n % d == 0:
            if d < n:
                divisors.add(d)
            other = n // d
            if other != d and other < n:
                divisors.add(other)
    s = set()
    for d in divisors:
        s.add(G[d])
    mex = 0
    while mex in s:
        mex += 1
    G[n] = mex

# Read input and determine the winner
n = int(input())
a = list(map(int, input().split()))
xor_sum = 0
for num in a:
    xor_sum ^= G[num]

print("Anna" if xor_sum != 0 else "Bruno")