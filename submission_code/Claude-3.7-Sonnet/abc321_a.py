# Read the input number N
N = input().strip()

# Check if each digit is strictly greater than the next digit
is_321_like = True
for i in range(len(N) - 1):
    if int(N[i]) <= int(N[i + 1]):
        is_321_like = False
        break

# Print the result
if is_321_like:
    print("Yes")
else:
    print("No")