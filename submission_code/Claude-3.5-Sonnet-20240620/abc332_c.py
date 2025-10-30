# YOUR CODE HERE
N, M = map(int, input().split())
S = input().strip()

logo_needed = 0
plain_needed = 0
logo_available = 0
plain_available = M

for day in S:
    if day == '0':
        logo_available += logo_needed
        plain_available += plain_needed
        logo_needed = 0
        plain_needed = 0
    elif day == '1':
        if logo_available + plain_available > 0:
            if plain_available > 0:
                plain_available -= 1
                plain_needed += 1
            else:
                logo_available -= 1
                logo_needed += 1
        else:
            plain_needed += 1
    elif day == '2':
        if logo_available > 0:
            logo_available -= 1
        else:
            logo_needed += 1
        logo_needed += 1

logo_to_buy = max(0, logo_needed - logo_available)
print(logo_to_buy)