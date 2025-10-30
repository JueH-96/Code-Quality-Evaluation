import math

def find_a_for_b(b):
    if b == 1:
        return 1
    for a in range(2, int(math.log2(b)) + 2):
        if a ** a == b:
            return a
    return -1

b = int(input().strip())
print(find_a_for_b(b))