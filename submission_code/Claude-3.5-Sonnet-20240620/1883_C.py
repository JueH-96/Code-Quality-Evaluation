# YOUR CODE HERE
def min_operations(n, k, arr):
    if k == 2:
        return 0 if any(x % 2 == 0 for x in arr) else 1
    elif k == 3:
        rem = sum(x % 3 for x in arr)
        return 0 if rem % 3 == 0 else 3 - (rem % 3)
    elif k == 4:
        rem2 = sum(x % 2 for x in arr)
        rem4 = sum(x % 4 for x in arr)
        if rem4 == 0:
            return 0
        elif rem2 == 0:
            return 1
        else:
            return 2 if rem4 <= 2 else 3 - (rem4 % 2)
    else:  # k == 5
        rem = sum(x % 5 for x in arr)
        return 0 if rem % 5 == 0 else 5 - (rem % 5)

t = int(input())
for _ in range(t):
    n, k = map(int, input().split())
    arr = list(map(int, input().split()))
    print(min_operations(n, k, arr))