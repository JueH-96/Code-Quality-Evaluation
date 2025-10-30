# YOUR CODE HERE
n = int(input())
s = list(input())
q = int(input())

for _ in range(q):
    query = input().split()
    type = int(query[0])
    if type == 1:
        index = int(query[1]) - 1
        char = query[2]
        s[index] = char
    elif type == 2:
        for i in range(n):
            if 'A' <= s[i] <= 'Z':
                s[i] = s[i].lower()
    elif type == 3:
        for i in range(n):
            if 'a' <= s[i] <= 'z':
                s[i] = s[i].upper()

print("".join(s))