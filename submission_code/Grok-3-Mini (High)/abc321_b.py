import sys
data = list(map(int, sys.stdin.read().split()))
N = data[0]
X = data[1]
A = data[2 : N + 1]  # List of the first N-1 scores
sum_A = sum(A)
min_A = min(A)
max_A = max(A)
max_possible = sum_A - min_A  # Maximum possible final grade

if X > max_possible:
    print(-1)  # Impossible to achieve X or higher
elif X <= sum_A - max_A:
    print(0)  # Minimum score needed is 0
else:
    S_needed = X - sum_A + min_A + max_A  # Calculate the needed score
    print(S_needed)