# YOUR CODE HERE
N, S, K = map(int, input().split())
prices = list(map(int, input().split()))
quantities = list(map(int, input().split()))

total_price = sum(p * q for p, q in zip(prices, quantities))
shipping_fee = max(0, total_price - S)
payment = total_price + shipping_fee
print(payment)