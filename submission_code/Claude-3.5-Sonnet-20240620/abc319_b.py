# YOUR CODE HERE
def generate_string(N):
    result = []
    for i in range(N + 1):
        divisor = None
        for j in range(1, 10):
            if N % j == 0 and i % (N // j) == 0:
                divisor = j
                break
        result.append(str(divisor) if divisor else '-')
    return ''.join(result)

N = int(input())
print(generate_string(N))