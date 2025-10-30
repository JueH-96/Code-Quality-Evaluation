s = input().strip()

while "WA" in s:
    index = s.find("WA")
    s = s[:index] + "AC" + s[index+2:]

print(s)