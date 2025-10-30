R, G, B = map(int, input().split())
C = input()

lst = [R, G, B]

if C == "Red":
  lst.remove(R)
elif C == "Green":
  lst.remove(G)
else:
  lst.remove(B)

print(min(lst))