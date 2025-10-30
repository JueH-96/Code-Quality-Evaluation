import sys

data = sys.stdin.read().split()
index = 0
t = int(data[index])
index += 1

for _ in range(t):
    n = int(data[index])
    index += 1
    a = list(map(int, data[index:index + n]))
    index += n
    
    # Compute num_zeros, min_val, prod_nonzero
    num_zeros = 0
    min_val = 10  # Initialize to a value larger than any digit
    prod_nonzero = 1
    for num in a:
        if num == 0:
            num_zeros += 1
        else:
            prod_nonzero *= num
            if num < min_val:
                min_val = num
    
    if num_zeros >= 2:
        print(0)
    elif num_zeros == 1:
        print(prod_nonzero)
    else:  # num_zeros == 0
        new_P = (prod_nonzero * (min_val + 1)) // min_val
        print(new_P)