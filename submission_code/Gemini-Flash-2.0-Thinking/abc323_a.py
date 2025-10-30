s = input()
indices_to_check = [1, 3, 5, 7, 9, 11, 13, 15]
all_zeros = True
for index in indices_to_check:
    if s[index] != '0':
        all_zeros = False
        break
if all_zeros:
    print("Yes")
else:
    print("No")