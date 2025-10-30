# YOUR CODE HERE
N = input().strip()

# Check if digits are strictly decreasing
is_321_like = True

for i in range(len(N) - 1):
    if int(N[i]) <= int(N[i + 1]):
        is_321_like = False
        break

if is_321_like:
    print("Yes")
else:
    print("No")