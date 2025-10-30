N = input()

if len(N) == 1:
    print("Yes")
else:
    is_321_like = True
    for i in range(len(N)-1):
        if N[i] <= N[i+1]:
            is_321_like = False
            break
    if is_321_like:
        print("Yes")
    else:
        print("No")