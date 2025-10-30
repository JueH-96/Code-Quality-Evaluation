def is_321_like(n):
    s = str(n)
    for i in range(len(s) - 1):
        if int(s[i]) <= int(s[i+1]):
            return False
    return True

def generate_321_like_numbers():
    result = []
    for i in range(1, 1024):
        num_str = ""
        for j in range(9, -1, -1):
            if (i >> (9 - j)) & 1:
                num_str += str(j)
        if num_str:
            result.append(int(num_str))
    result.sort()
    return result

k = int(input())
numbers = generate_321_like_numbers()
print(numbers[k-1])