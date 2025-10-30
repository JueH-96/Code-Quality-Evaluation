def min_tshirts_to_buy(N, M, S):
    logo_tshirts_needed = 0
    plain_tshirts = M
    logo_tshirts = 0

    for day in S:
        if day == '0':
            # Wash all T-shirts
            plain_tshirts += logo_tshirts
            logo_tshirts = 0
        elif day == '1':
            if plain_tshirts > 0:
                plain_tshirts -= 1
            else:
                logo_tshirts -= 1
        elif day == '2':
            if logo_tshirts > 0:
                logo_tshirts -= 1
            else:
                logo_tshirts_needed += 1
                logo_tshirts = 1

    return logo_tshirts_needed

# Read input from stdin
N, M = map(int, input().split())
S = input().strip()

# Solve the problem and print the result to stdout
print(min_tshirts_to_buy(N, M, S))