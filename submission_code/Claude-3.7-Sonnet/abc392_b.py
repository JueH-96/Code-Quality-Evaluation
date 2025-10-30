# Read input
N, M = map(int, input().split())
A = set(map(int, input().split()))

# Find missing integers between 1 and N
all_integers = set(range(1, N + 1))
missing_integers = sorted(all_integers - A)

# Output the count of missing integers
print(len(missing_integers))

# Output the missing integers if there are any
if missing_integers:
    print(' '.join(map(str, missing_integers)))