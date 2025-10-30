import sys

s = input().strip()

if s[:3] == "ABC" and s[3:].isdigit():
    contest_num = int(s[3:])
    if 1 <= contest_num <= 315 or 317 <= contest_num <= 348:
        print("Yes")
    else:
        print("No")
else:
    print("No")