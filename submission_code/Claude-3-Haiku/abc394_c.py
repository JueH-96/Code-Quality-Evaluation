import sys

def replace_wa(s):
    result = ""
    i = 0
    while i < len(s):
        if i + 1 < len(s) and s[i:i+2] == "WA":
            result += "AC"
            i += 2
        else:
            result += s[i]
            i += 1
    return result

s = input()
print(replace_wa(s))