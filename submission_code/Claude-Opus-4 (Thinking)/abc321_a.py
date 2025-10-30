N = input()

is_321_like = True
for i in range(len(N) - 1):
    if N[i] <= N[i+1]:
        is_321_like = False
        break

print("Yes" if is_321_like else "No")