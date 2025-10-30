s = input().strip()
t = input().strip()

i = len(s) - 1
j = len(t) - 1
result = []

while i >= 0 and j >= 0:
    if s[i] == t[j]:
        result.append(j + 1)  # positions are 1-based
        i -= 1
        j -= 1
    else:
        j -= 1

# Reverse the result to get the correct order
result.reverse()
print(' '.join(map(str, result)))