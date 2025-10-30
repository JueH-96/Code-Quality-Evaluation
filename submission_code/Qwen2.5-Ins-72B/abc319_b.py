# YOUR CODE HERE
def find_smallest_divisor(N, i):
    for j in range(1, 10):
        if N % j == 0 and i % (N // j) == 0:
            return str(j)
    return '-'

N = int(input())
result = ''.join(find_smallest_divisor(N, i) for i in range(N + 1))
print(result)