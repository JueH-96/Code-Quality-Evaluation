import sys

# Read N from stdin
N = int(sys.stdin.readline().strip())

# Compute the minimum M such that 2^M >= N
M = 0
while (1 << M) < N:
    M += 1

# Output M and flush
print(M)
sys.stdout.flush()

# For each friend k from 1 to M, output the distribution
for k in range(1, M + 1):
    bottles = []
    for j in range(1, N + 1):
        num = j - 1
        if (num >> (M - k)) & 1:
            bottles.append(j)
    # K is the number of bottles, output K and the bottles in ascending order
    K_len = len(bottles)
    line = [str(K_len)] + [str(bottle) for bottle in bottles]
    print(' '.join(line))

# Flush after all distributions
sys.stdout.flush()

# Read the string S from stdin
S = sys.stdin.readline().strip()

# Compute the spoiled bottle number
num_spoiled = int(S, 2)
X = num_spoiled + 1

# Output X and flush
print(X)
sys.stdout.flush()