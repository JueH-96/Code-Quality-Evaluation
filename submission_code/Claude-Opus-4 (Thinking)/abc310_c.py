# YOUR CODE HERE
n = int(input())
canonical_forms = set()

for _ in range(n):
    s = input()
    reversed_s = s[::-1]
    # Take the lexicographically smaller one as canonical form
    canonical = min(s, reversed_s)
    canonical_forms.add(canonical)

print(len(canonical_forms))