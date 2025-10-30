n_str = input()
n = len(n_str)

if n == 1:
    print("Yes")
else:
    is_321_like = True
    for i in range(n - 1):
        if int(n_str[i]) <= int(n_str[i+1]):
            is_321_like = False
            break
    if is_321_like:
        print("Yes")
    else:
        print("No")