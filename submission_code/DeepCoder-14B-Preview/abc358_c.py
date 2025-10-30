import itertools

# Read input
n, m = map(int, input().split())
stands = []
for _ in range(n):
    s = input().strip()
    current = set()
    for j in range(m):
        if s[j] == 'o':
            current.add(j + 1)  # Convert to 1-based flavor index
    stands.append(current)

# Check all possible combinations of stands
for k in range(1, n + 1):
    for combo in itertools.combinations(stands, k):
        combined = set()
        for c in combo:
            combined.update(c)
        if len(combined) == m:
            print(k)
            exit()

# The problem guarantees a solution, so this line is theoretically unreachable
print(n)