N = int(input())
rates = []
for i in range(N):
    a, b = map(int, input().split())
    # Store tuple of (success rate, person number)
    # Multiply by -1 to handle ties in ascending order of numbers
    rates.append((-a/(a+b), -(i+1)))

# Sort by success rate (descending) and person number (ascending)
rates.sort()

# Print person numbers in required order
print(*[-x[1] for x in rates])