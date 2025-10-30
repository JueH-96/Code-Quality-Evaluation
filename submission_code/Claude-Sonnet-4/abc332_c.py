# YOUR CODE HERE
N, M = map(int, input().split())
S = input().strip()

logo_shirts_needed = 0
available_plain = M
available_logo = 0

for i in range(N):
    if S[i] == '0':
        # Wash all T-shirts - reset available counts
        available_plain = M
        available_logo = logo_shirts_needed
    elif S[i] == '1':
        # Need any T-shirt (plain or logo)
        if available_plain > 0:
            available_plain -= 1
        elif available_logo > 0:
            available_logo -= 1
        else:
            # Need to buy a logo T-shirt
            logo_shirts_needed += 1
            available_logo = logo_shirts_needed - 1
    elif S[i] == '2':
        # Must wear logo T-shirt
        if available_logo > 0:
            available_logo -= 1
        else:
            # Need to buy a logo T-shirt
            logo_shirts_needed += 1
            available_logo = logo_shirts_needed - 1

print(logo_shirts_needed)