s = input().strip()

# Determine the majority character
if s[0] == s[1] or s[0] == s[2]:
    majority = s[0]
else:
    majority = s[1]

# Find the unique character's position
for i in range(len(s)):
    if s[i] != majority:
        print(i + 1)
        break