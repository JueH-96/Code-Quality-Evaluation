# YOUR CODE HERE
N = int(input())

if N % 2 == 1:  # odd length
    # One = at the center
    center = N // 2
    result = ['-'] * N
    result[center] = '='
else:  # even length
    # Two =s at the center
    center1 = N // 2 - 1
    center2 = N // 2
    result = ['-'] * N
    result[center1] = '='
    result[center2] = '='

print(''.join(result))