def remove_abc(s):
    while "ABC" in s:
        idx = s.find("ABC")
        s = s[:idx] + s[idx+3:]
    return s

S = input().strip()
print(remove_abc(S))