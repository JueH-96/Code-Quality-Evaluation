n = int(input())
s = input().strip()

has_a = False
has_b = False
has_c = False

for i in range(n):
    char = s[i]
    if char == 'A':
        has_a = True
    elif char == 'B':
        has_b = True
    else:
        has_c = True
    if has_a and has_b and has_c:
        print(i + 1)
        break