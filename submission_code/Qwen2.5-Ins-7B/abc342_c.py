# YOUR CODE HERE
n = int(input())
s = input()
q = int(input())
replacements = {}
for _ in range(q):
    c, d = input().split()
    replacements[c] = d

for c, d in replacements.items():
    s = s.replace(c, d)

print(s)