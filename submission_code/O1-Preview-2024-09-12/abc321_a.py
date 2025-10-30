N = input().strip()
digits = [int(c) for c in N]

is_321_like = True

for i in range(len(digits)-1):
    if digits[i] <= digits[i+1]:
        is_321_like = False
        break

if is_321_like:
    print("Yes")
else:
    print("No")