# YOUR CODE HERE
n_str = input()
n_list = list(n_str)
is_321_like = True
for i in range(len(n_list) - 1):
    if int(n_list[i]) <= int(n_list[i+1]):
        is_321_like = False
        break
if is_321_like:
    print("Yes")
else:
    print("No")