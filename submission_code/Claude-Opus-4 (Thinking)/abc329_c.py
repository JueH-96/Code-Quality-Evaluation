N = int(input())
S = input()

max_lengths = {}

i = 0
while i < N:
    char = S[i]
    j = i
    # Count consecutive occurrences of the current character
    while j < N and S[j] == char:
        j += 1
    length = j - i
    # Update maximum length for this character
    max_lengths[char] = max(max_lengths.get(char, 0), length)
    i = j

# Sum all maximum lengths
total = sum(max_lengths.values())
print(total)