# NUMBER OF MAIN DISHES, SIDE DISHES AND P
N, M, P = map(int, input().split())

# PRICES OF MAIN DISHES
A = list(map(int, input().split()))

# PRICES OF SIDE DISHES
B = list(map(int, input().split()))

# SORT THE PRICES
A.sort()
B.sort()

# TOTAL PRICE
total_price = 0

# CALCULATE THE TOTAL PRICE OF ALL SET MEALS
i, j = 0, M - 1
while i < N and j >= 0:
    if A[i] + B[j] <= P:
        total_price += (P * (j + 1))
        i += 1
    else:
        total_price += (A[i] + B[j])
        j -= 1

# ADD THE REMAINING SIDE DISHES
total_price += (B[j + 1:] + [0] * (N - i))[:N - i]

print(total_price)