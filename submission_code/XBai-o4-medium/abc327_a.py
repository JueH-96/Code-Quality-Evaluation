n = int(input())
s = input()

found = False
for i in range(n - 1):
    if {s[i], s[i + 1]} == {'a', 'b'}:
        found = True
        break

print("Yes" if found else "No")