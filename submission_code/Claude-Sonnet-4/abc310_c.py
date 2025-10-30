# YOUR CODE HERE
n = int(input())
unique_sticks = set()

for _ in range(n):
    s = input().strip()
    # Create canonical form by taking lexicographically smaller of s and its reverse
    canonical = min(s, s[::-1])
    unique_sticks.add(canonical)

print(len(unique_sticks))