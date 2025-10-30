# YOUR CODE HERE
import sys

def solve():
    # Read the first line containing N, S, K
    n, s, k = map(int, sys.stdin.readline().split())

    # Initialize the total price of the products
    total_product_price = 0

    # Loop N times to read product details and calculate the total price
    for _ in range(n):
        # Read the price P_i and quantity Q_i for the current product type
        p, q = map(int, sys.stdin.readline().split())
        # Add the cost of this product type (price * quantity) to the total
        total_product_price += p * q

    # Determine the shipping fee based on the total product price
    shipping_fee = 0
    # If the total price is less than the threshold S, apply the shipping fee K
    if total_product_price < s:
        shipping_fee = k

    # Calculate the final amount Takahashi will pay
    # Final amount = total price of products + shipping fee
    final_amount = total_product_price + shipping_fee

    # Print the final amount
    print(final_amount)

# Execute the solve function to run the program
solve()