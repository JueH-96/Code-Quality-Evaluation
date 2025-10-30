def main():
    # Read input values
    N, P, Q = map(int, input().split())
    dishes = list(map(int, input().split()))
    
    # Calculate the minimum cost if using the discount coupon:
    # The coupon is only applicable when one orders an additional dish.
    cost_with_coupon = Q + min(dishes)
    
    # The answer is the minimum of the regular price and the discounted combination
    answer = min(P, cost_with_coupon)
    
    # Print the answer
    print(answer)

# Call the main function
if __name__ == "__main__":
    main()