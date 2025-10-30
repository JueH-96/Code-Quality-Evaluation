# YOUR CODE HERE
import sys
from math import ceil, log2

N = int(sys.stdin.readline())

M = ceil(log2(N))
M = int(M)
print(M)
sys.stdout.flush()

# Initialize friends' bottle lists
friends = [[] for _ in range(M)]  # friends[0] corresponds to bit position 0

for b in range(1, N+1):
    b_bits = b - 1
    for bit_pos in range(M):
        if (b_bits >> bit_pos) & 1:
            friends[bit_pos].append(b)

# Output the assignments
for i in range(M):
    bottles = sorted(friends[i])
    Ki = len(bottles)
    print(Ki, *bottles)
    sys.stdout.flush()

# Read the string S
S = sys.stdin.readline().strip()

X_prime = 0
for i in range(M):
    if S[i] == '1':
        X_prime |= 1 << i
X_prime += 1  # Adjust to match bottle number starting from 1

# Ensure X_prime is within 1 to N
if X_prime < 1 or X_prime > N:
    X_prime = 1  # Default to 1 if out of range

print(X_prime)
sys.stdout.flush()