# YOUR CODE HERE
s = input()
t = input()

n = len(s)
m = len(t)

if m == 3:
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                if s[i].upper() == t[0] and s[j].upper() == t[1] and s[k].upper() == t[2]:
                    print("Yes")
                    exit()

if m == 3:
    for i in range(n):
        for j in range(i + 1, n):
            if s[i].upper() == t[0] and s[j].upper() == t[1] and t[2] == 'X':
                print("Yes")
                exit()

print("No")