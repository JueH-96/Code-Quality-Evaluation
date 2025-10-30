s = input()
presses = 0
i = 0

while i < len(s):
    if i < len(s) - 1 and s[i] == '0' and s[i+1] == '0':
        presses += 1
        i += 2
    else:
        presses += 1
        i += 1

print(presses)