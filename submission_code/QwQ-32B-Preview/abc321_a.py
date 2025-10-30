def is_321_like_number(N_str):
    if len(N_str) == 1:
        return "Yes"
    for i in range(len(N_str) - 1):
        if int(N_str[i]) <= int(N_str[i + 1]):
            return "No"
    return "Yes"

# Read input
N = input().strip()

# Get and print the result
result = is_321_like_number(N)
print(result)