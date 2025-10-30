# YOUR CODE HERE
def approximate_number(N):
    if N <= 999:
        return N
    elif 1000 <= N <= 9999:
        return N // 10 * 10
    elif 10000 <= N <= 99999:
        return N // 100 * 100
    elif 100000 <= N <= 999999:
        return N // 1000 * 1000
    elif 1000000 <= N <= 9999999:
        return N // 10000 * 10000
    elif 10000000 <= N <= 99999999:
        return N // 100000 * 100000
    else:
        return N // 1000000 * 1000000

N = int(input())
print(approximate_number(N))