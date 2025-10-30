def is_321_like_number(N):
    N_str = str(N)
    for i in range(len(N_str) - 1):
        if N_str[i] <= N_str[i + 1]:
            return "No"
    return "Yes"

N = int(input())
print(is_321_like_number(N))