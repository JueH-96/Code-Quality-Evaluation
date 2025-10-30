def main():
    import sys
    input_data = sys.stdin.read().split()
    
    # The first three inputs are the prices for red, green, and blue pens.
    R, G, B = map(int, input_data[:3])
    # The fourth input is the disliked color.
    disliked_color = input_data[3].strip()
    
    # Create a dictionary mapping colors to their prices.
    prices = {"Red": R, "Green": G, "Blue": B}
    
    # Remove the color Takahashi dislikes.
    del prices[disliked_color]
    
    # The answer is the minimum price among the remaining pen options.
    print(min(prices.values()))
    
# Don't forget to call main() as required.
if __name__ == '__main__':
    main()