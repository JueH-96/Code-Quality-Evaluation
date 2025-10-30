n, d = map(int, input().split())
s = input().strip()

# Collect positions of '@' in the string
at_positions = [i for i, c in enumerate(s) if c == '@']
# Sort in reverse order to get the rightmost positions first
at_positions.sort(reverse=True)
# Take the first D positions to be eaten
eaten = at_positions[:d]
eaten_set = set(eaten)

# Build the result string
result = []
for i in range(n):
    if s[i] == '@' and i in eaten_set:
        result.append('.')
    else:
        result.append(s[i])

print(''.join(result))