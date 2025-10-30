# YOUR CODE HERE

def find_smallest_divisor(n):
    for i in range(1, 10):
        if n % i == 0:
            return i
    return -1

def generate_string(n):
    result = ""
    for i in range(n+1):
        if i % n == 0:
            result += str(find_smallest_divisor(i))
        else:
            result += "-"
    return result

n = int(input())
print(generate_string(n))