s = input().strip()

# Determine the majority character
if s[0] == s[1]:
    majority = s[0]
else:
    if s[2] == s[0]:
        majority = s[0]
    else:
        majority = s[1]

# Find and print the position of the differing character
for i in range(len(s)):
    if s[i] != majority:
        print(i + 1)
        break