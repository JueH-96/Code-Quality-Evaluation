R, G, B = map(int, input().split())
C = input()
if C == "Red":
    available = [G, B]
elif C == "Green":
    available = [R, B]
elif C == "Blue":
    available = [R, G]
print(min(available))