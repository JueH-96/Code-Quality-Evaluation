# Read the inputs
N, M, P = map(int, input().split())
A = list(map(int, input().split()))
B = list(map(int, input().split()))

# Calculate the total price of all set meals
total_price = 0
for main_dish in A:
    for side_dish in B:
        total_price += min(main_dish + side_dish, P)

# Print the answer
print(total_price)