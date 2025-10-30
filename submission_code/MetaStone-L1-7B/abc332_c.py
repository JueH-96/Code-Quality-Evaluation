n, m = map(int, input().split())
s = input().strip()

available_plain = m
available_logo = 0
bought = 0

used_plain = [0] * n
used_logo = [0] * n

for i in range(n):
    day = s[i]
    if day == '2':
        if available_logo > 0:
            available_logo -= 1
        else:
            bought += 1
            available_logo += 1
        used_logo[i] += 1
    elif day == '1':
        if available_plain > 0:
            available_plain -= 1
        else:
            bought += 1
            available_logo += 1
        used_plain[i] += 1
    elif day == '0':
        available_plain += used_plain[i]
        available_logo += used_logo[i]
        used_plain[i] = 0
        used_logo[i] = 0

print(bought)