import sys
s = sys.stdin.read().strip()
insertions = 0
exp = 'i'
for char in s:
    while exp != char:
        insertions += 1
        if exp == 'i':
            exp = 'o'
        else:
            exp = 'i'
    # After matching the character, toggle the expectation
    if exp == 'i':
        exp = 'o'
    else:
        exp = 'i'
# After processing all characters, check if we need to insert 'o'
if exp == 'o':
    insertions += 1
print(insertions)