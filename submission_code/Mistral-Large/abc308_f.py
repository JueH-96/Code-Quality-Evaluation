import sys
input = sys.stdin.read

def solve():
    data = input().split()
    index = 0

    N = int(data[index])
    M = int(data[index + 1])
    index += 2

    P = [int(data[index + i]) for i in range(N)]
    index += N

    L = [int(data[index + i]) for i in range(M)]
    index += M

    D = [int(data[index + i]) for i in range(M)]

    # Create a list of tuples (L_i, D_i)
    coupons = list(zip(L, D))

    # Sort coupons based on L_i in descending order
    coupons.sort(reverse=True, key=lambda x: x[0])

    # Sort items based on P_i in descending order
    P.sort(reverse=True)

    total_cost = 0
    coupon_index = 0

    # Iterate through each item
    for price in P:
        # If there are coupons left and the current coupon can be applied to the item
        if coupon_index < M and coupons[coupon_index][0] <= price:
            # Apply the coupon
            total_cost += price - coupons[coupon_index][1]
            coupon_index += 1
        else:
            # Buy the item at the regular price
            total_cost += price

    print(total_cost)

if __name__ == "__main__":
    solve()