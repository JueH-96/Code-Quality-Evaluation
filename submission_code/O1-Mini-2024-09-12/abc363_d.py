N = int(input())
l = 1
while True:
    if l == 1:
        cnt = 10
    elif l % 2 == 0:
        k = l // 2
        cnt = 9 * (10 ** (k - 1))
    else:
        k = (l // 2) + 1
        cnt = 9 * (10 ** (k - 1))
    if N <= cnt:
        break
    N -= cnt
    l += 1
if l == 1:
    palindrome = N - 1
elif l % 2 == 0:
    k = l // 2
    base = 10 ** (k - 1)
    first_half = base + (N - 1)
    first_half_str = str(first_half)
    palindrome = int(first_half_str + first_half_str[::-1])
else:
    k = (l // 2) + 1
    base = 10 ** (k - 1)
    first_half = base + (N - 1)
    first_half_str = str(first_half)
    palindrome = int(first_half_str + first_half_str[:-1][::-1])
print(palindrome)