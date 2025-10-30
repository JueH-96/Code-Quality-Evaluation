n = input()
is_321_like = True
for i in range(len(n) - 1):
    if int(n[i]) <= int(n[i+1]):
        is_321_like = False
        break

if is_321_like:
    print("Yes")
else:
    print("No")