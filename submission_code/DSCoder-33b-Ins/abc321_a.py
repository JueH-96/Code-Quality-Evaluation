# YOUR CODE HERE

N = int(input())

def is_321_like(n):
    str_n = str(n)
    for i in range(len(str_n) - 1):
        if str_n[i] <= str_n[i + 1]:
            return "No"
    return "Yes"

print(is_321_like(N))