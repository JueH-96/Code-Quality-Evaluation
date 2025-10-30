N = int(input())

# Find all divisors j of N where 1 <= j <= 9
divisors = [j for j in range(1, 10) if N % j == 0]

# Sort the divisors in ascending order
divisors.sort()

# Initialize the result string
result = ''

# For each i from 0 to N
for i in range(N + 1):
    found = False
    # Check each j in ascending order
    for j in divisors:
        d = N // j
        if i % d == 0:
            result += str(j)
            found = True
            break
    if not found:
        result += '-'

print(result)