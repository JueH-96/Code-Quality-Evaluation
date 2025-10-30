from collections import defaultdict

N = int(input().strip())
S = [input().strip() for _ in range(N)]

# Create a dictionary to store the reversed strings
reversed_strings = defaultdict(list)

for i in range(N):
    reversed_strings[S[i]].append(i)
    reversed_strings[S[i][::-1]].append(i)

# Remove duplicates and count the number of unique sticks
unique_sticks = set()

for key in reversed_strings:
    if len(reversed_strings[key]) > 1:
        for i in reversed_strings[key]:
            unique_sticks.add(i)

print(len(unique_sticks))