# Read input
N, Q = map(int, input().split())
treatments = list(map(int, input().split()))

# Track holes that have undergone treatment an odd number of times
odd_treatments = set()

for t in treatments:
    if t in odd_treatments:
        odd_treatments.remove(t)
    else:
        odd_treatments.add(t)

# Final tooth count is N minus the number of holes that have undergone an odd number of treatments
print(N - len(odd_treatments))