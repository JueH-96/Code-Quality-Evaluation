n = int(input())
s = input().strip()
t = input().strip()

for i in range(n):
    sc = s[i]
    tc = t[i]
    if sc == tc:
        continue
    # Check group 0 and o
    if sc in {'0', 'o'} and tc in {'0', 'o'}:
        continue
    # Check group 1 and l
    if sc in {'1', 'l'} and tc in {'1', 'l'}:
        continue
    # If none of the conditions are met
    print("No")
    exit()

print("Yes")