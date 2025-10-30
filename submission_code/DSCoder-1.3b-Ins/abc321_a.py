# YOUR CODE HERE

def is_321_like(n):
    str_n = str(n)
    for i in range(len(str_n) - 2):
        if str_n[i] > str_n[i+1]:
            return "No"
    return "Yes"

n = int(input())
print(is_321_like(n))