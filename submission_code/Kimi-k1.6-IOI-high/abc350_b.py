n, q = map(int, input().split())
treatments = list(map(int, input().split()))

# Initialize teeth status: True means has a tooth, False means no tooth
teeth = [True] * (n + 1)  # 1-based indexing, 0 is unused

for t in treatments:
    teeth[t] = not teeth[t]

# Count the number of True values from index 1 to n
print(sum(teeth[1:n+1]))