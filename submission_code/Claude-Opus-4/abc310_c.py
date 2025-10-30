# YOUR CODE HERE
n = int(input())
unique_sticks = set()

for _ in range(n):
    s = input().strip()
    # For each string, use the lexicographically smaller one between itself and its reverse
    # This ensures that "abc" and "cba" are treated as the same stick
    normalized = min(s, s[::-1])
    unique_sticks.add(normalized)

print(len(unique_sticks))