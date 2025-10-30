N, K, X = map(int, input().split())
T = list(map(int, input().split()))

orders = []
for i in range(N):
    orders.append((T[i], i))

orders.sort()

shipments = []
total_dissatisfaction = 0
next_shipment_day = 0

i = 0
while i < N:
    current_shipment = []
    current_shipment_day = max(next_shipment_day, orders[i][0])
    count = 0
    j = i
    while j < N and count < K and orders[j][0] <= current_shipment_day:
        current_shipment.append(orders[j])
        count += 1
        j += 1
    
    for order in current_shipment:
        total_dissatisfaction += current_shipment_day - order[0]

    next_shipment_day = current_shipment_day + X
    i = j

print(total_dissatisfaction)