# Read the input
N, M, P = map(int, input().split())
A = list(map(int, input().split()))
B = list(map(int, input().split()))

# Calculate the total price of all set meals
total_price = 0
for a in A:
    for b in B:
        total_price += min(a + b, P)

# Print the answer
print(total_price)