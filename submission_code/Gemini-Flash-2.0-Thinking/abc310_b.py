def solve():
    n, m = map(int, input().split())
    products = []
    for _ in range(n):
        line = list(map(int, input().split()))
        price = line[0]
        count = line[1]
        functions = set(line[2:])
        products.append((price, functions))

    for i in range(n):
        for j in range(n):
            if i == j:
                continue

            price_i, functions_i = products[i]
            price_j, functions_j = products[j]

            condition1 = price_i >= price_j
            condition2 = functions_i.issubset(functions_j)
            condition3 = price_i > price_j or len(functions_j - functions_i) > 0

            if condition1 and condition2 and condition3:
                print("Yes")
                return

    print("No")

solve()