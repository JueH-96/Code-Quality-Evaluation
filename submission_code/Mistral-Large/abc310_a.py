import sys

input = sys.stdin.read
data = input().split()

N = int(data[0])
P = int(data[1])
Q = int(data[2])
D = list(map(int, data[3:]))

min_dish_price = min(D)

total_with_coupon = Q + min_dish_price
total_without_coupon = P

answer = min(total_with_coupon, total_without_coupon)

print(answer)