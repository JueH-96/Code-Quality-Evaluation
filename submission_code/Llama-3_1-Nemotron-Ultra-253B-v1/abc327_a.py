n = int(input())
s = input().strip()

for i in range(len(s) - 1):
    if {s[i], s[i+1]} == {'a', 'b'}:
        print("Yes")
        exit()
print("No")