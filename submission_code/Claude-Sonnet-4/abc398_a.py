# YOUR CODE HERE
N = int(input())

if N % 2 == 1:
    # Odd length: place single '=' in the center
    result = '-' * (N // 2) + '=' + '-' * (N // 2)
else:
    # Even length: place two adjacent '=' in the center
    result = '-' * (N // 2 - 1) + '==' + '-' * (N // 2 - 1)

print(result)