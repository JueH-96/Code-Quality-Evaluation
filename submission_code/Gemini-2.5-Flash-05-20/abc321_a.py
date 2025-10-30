N_str = input()

is_321_like = True
# Iterate through the digits from the first to the second-to-last.
# For each digit at index 'i', compare it with the digit at index 'i+1'.
for i in range(len(N_str) - 1):
    current_digit = int(N_str[i])
    next_digit = int(N_str[i+1])

    # If the current digit is not strictly greater than the next digit,
    # then N is not a 321-like Number.
    if current_digit <= next_digit:
        is_321_like = False
        break # No need to check further, we found a violation

if is_321_like:
    print("Yes")
else:
    print("No")