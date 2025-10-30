import sys

def min_tshirts_to_buy(N, M, S):
    logo_tshirts_needed = 0
    available_tshirts = M
    for day in S:
        if day == '1':
            if available_tshirts > 0:
                available_tshirts -= 1
            else:
                logo_tshirts_needed += 1
        elif day == '2':
            if available_tshirts > 0:
                available_tshirts -= 1
            else:
                logo_tshirts_needed += 1
        elif day == '0':
            available_tshirts += logo_tshirts_needed
            logo_tshirts_needed = 0
    return logo_tshirts_needed

input = sys.stdin.read
data = input().split()

N = int(data[0])
M = int(data[1])
S = data[2]

print(min_tshirts_to_buy(N, M, S))