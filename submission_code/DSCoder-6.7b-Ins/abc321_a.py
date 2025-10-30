def is_321_like(n):
    n = str(n)
    for i in range(len(n) - 1):
        if n[i] < n[i+1]:
            return False
    return True

N = int(input())
print("Yes" if is_321_like(N) else "No")