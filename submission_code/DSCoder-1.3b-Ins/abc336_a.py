# YOUR CODE HERE

def dragon_string(n):
    if n == 1:
        return 'L'
    else:
        return 'L' + 'o' * n + 'n' + 'g' * (n + 1)

n = int(input())
print(dragon_string(n))