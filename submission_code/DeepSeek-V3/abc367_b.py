# YOUR CODE HERE
X = input().strip()
if '.' in X:
    integer_part, decimal_part = X.split('.')
    decimal_part = decimal_part.rstrip('0')
    if decimal_part:
        print(f"{integer_part}.{decimal_part}")
    else:
        print(integer_part)
else:
    print(X)