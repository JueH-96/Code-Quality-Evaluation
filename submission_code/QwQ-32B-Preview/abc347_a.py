# Read the first line and split into N and K
N_K = input().split()
N = int(N_K[0])
K = int(N_K[1])

# Read the second line and split into list of integers A
A = list(map(int, input().split()))

# Find multiples of K, divide by K, and collect quotients
quotients = [a // K for a in A if a % K == 0]

# Sort the quotients in ascending order
sorted_quotients = sorted(quotients)

# Print the sorted quotients separated by spaces
print(' '.join(map(str, sorted_quotients)))