n = int(input())
unique_sticks = set()

for _ in range(n):
    s = input()
    reverse_s = s[::-1]
    normalized = min(s, reverse_s)
    unique_sticks.add(normalized)

print(len(unique_sticks))