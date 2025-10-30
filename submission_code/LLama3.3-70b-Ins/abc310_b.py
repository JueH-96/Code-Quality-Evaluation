import sys

def main():
    N, M = map(int, sys.stdin.readline().split())
    products = []
    for _ in range(N):
        product = list(map(int, sys.stdin.readline().split()))
        price = product[0]
        functions = set(product[2:])
        products.append((price, functions))

    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            price_i, functions_i = products[i]
            price_j, functions_j = products[j]
            if price_i >= price_j and functions_i.issubset(functions_j) and (price_i > price_j or not functions_i == functions_j):
                print("Yes")
                return
    print("No")

if __name__ == "__main__":
    main()