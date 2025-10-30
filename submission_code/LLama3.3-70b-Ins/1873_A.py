import sys

def solve(s):
    if s == "abc":
        return "YES"
    for i in range(3):
        for j in range(i+1, 3):
            new_s = list(s)
            new_s[i], new_s[j] = new_s[j], new_s[i]
            if "".join(new_s) == "abc":
                return "YES"
    return "NO"

t = int(input())
for _ in range(t):
    s = input()
    print(solve(s))