s = input()
while True:
    index = s.find("WA")
    if index == -1:
        break
    s = s[:index] + "AC" + s[index+2:]
print(s)