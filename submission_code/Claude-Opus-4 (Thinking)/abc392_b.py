# YOUR CODE HERE
# Read input
N, M = map(int, input().split())
A = list(map(int, input().split()))

# Find missing integers
all_integers = set(range(1, N + 1))
A_set = set(A)
missing_integers = sorted(all_integers - A_set)

# Output
print(len(missing_integers))
if missing_integers:
    print(' '.join(map(str, missing_integers)))