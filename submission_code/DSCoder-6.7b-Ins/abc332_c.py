def min_tshirts(N, M, S):
    plain_tshirts = M
    logo_tshirts = 0
    for i in range(N):
        if S[i] == '1':
            if plain_tshirts == 0:
                logo_tshirts += 1
            else:
                plain_tshirts -= 1
        elif S[i] == '2':
            logo_tshirts += 1
        else:
            plain_tshirts = logo_tshirts
            logo_tshirts = 0
    return logo_tshirts

N, M = map(int, input().split())
S = input()
print(min_tshirts(N, M, S))