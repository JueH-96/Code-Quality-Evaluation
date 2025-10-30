# YOUR CODE HERE

def truncate_number(n):
    if n <= 10**3 - 1:
        return n
    elif 10**3 <= n < 10**4:
        return n // 10 * 10
    elif 10**4 <= n < 10**5:
        return n // 100 * 100
    elif 10**5 <= n < 10**6:
        return n // 1000 * 1000
    elif 10**6 <= n < 10**7:
        return n // 10000 * 10000
    elif 10**7 <= n < 10**8:
        return n // 100000 * 100000
    elif 10**8 <= n < 10**9:
        return n // 1000000 * 1000000

n = int(input())
print(truncate_number(n))