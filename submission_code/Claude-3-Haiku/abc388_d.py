# Read the input
N = int(input())
A = list(map(int, input().split()))

# Initialize the number of stones for each alien
B = [0] * N

# Simulate the process for N years
for i in range(N):
    # Give stones to the alien who just became an adult
    for j in range(i):
        if A[j] > 0:
            A[j] -= 1
            B[i] += 1

    # Update the number of stones for the current alien
    B[i] += A[i]

# Print the final number of stones for each alien
print(" ".join(map(str, B)))