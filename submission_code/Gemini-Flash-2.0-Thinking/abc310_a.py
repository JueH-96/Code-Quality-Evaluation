n, p, q = map(int, input().split())
d = list(map(int, input().split()))

cost_option1 = p
cost_option2 = q + min(d)

min_cost = min(cost_option1, cost_option2)
print(min_cost)