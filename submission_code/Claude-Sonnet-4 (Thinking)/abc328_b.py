def get_repdigit_digit(n):
    s = str(n)
    if len(set(s)) == 1:
        return int(s[0])
    else:
        return None

n = int(input())
d = list(map(int, input().split()))

# Precompute repdigits for each digit
repdigits_by_digit = {i: [] for i in range(1, 10)}
for num in range(1, 101):
    digit = get_repdigit_digit(num)
    if digit is not None:
        repdigits_by_digit[digit].append(num)

count = 0
for i in range(1, n + 1):
    month_digit = get_repdigit_digit(i)
    if month_digit is None:
        continue
    
    for repdigit in repdigits_by_digit[month_digit]:
        if repdigit <= d[i - 1]:
            count += 1

print(count)