import sys

# Read the number of countries
n = int(sys.stdin.readline())

# Read the initial amounts of currency for each country.
# We use a list `a` with 0-based indexing.
a = list(map(int, sys.stdin.readline().split()))

# Process exchanges sequentially for each country from 1 to N-1.
# In 0-based indexing, this corresponds to indices 0 to n-2.
for i in range(n - 1):
    # Read the exchange parameters: S_i units of country i for T_i units of country i+1.
    s, t = map(int, sys.stdin.readline().split())
    
    # It is always optimal to convert as much currency as possible at each step.
    # Calculate the number of times the exchange from country i to i+1 can be performed.
    num_exchanges = a[i] // s
    
    # Update the amount of currency for the next country, i+1.
    a[i+1] += num_exchanges * t

# After all possible exchanges are made, the amount for the last country (N) is maximized.
# Print the final amount for country N (at index n-1).
print(a[n-1])