# YOUR CODE HERE
def kth_321_like_number(k):
    k -= 1
    digits = []
    for i in range(9, -1, -1):
        while k >= 9 - i:
            k -= 9 - i
            digits.append(i)
        if k == 0:
            break
    return int(''.join(map(str, digits)))

k = int(input())
print(kth_321_like_number(k))