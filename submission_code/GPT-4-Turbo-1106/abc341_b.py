def max_currency(N, currencies, exchanges):
    for i in range(N - 2, -1, -1):
        S, T = exchanges[i]
        max_exchange = currencies[i] // S
        currencies[i] -= max_exchange * S
        currencies[i + 1] += max_exchange * T
    return currencies[-1]

# Read input from stdin
N = int(input().strip())
currencies = list(map(int, input().strip().split()))
exchanges = [tuple(map(int, input().strip().split())) for _ in range(N - 1)]

# Calculate and print the answer
print(max_currency(N, currencies, exchanges))