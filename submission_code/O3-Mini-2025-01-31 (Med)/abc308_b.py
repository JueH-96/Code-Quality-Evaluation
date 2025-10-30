def main():
    import sys
    input_data = sys.stdin.read().split()
    # input_data now contains all tokens in order
    # tokens: first token is N, second is M, then N sushi colors, then M special color names, 
    # then M+1 prices (P0 and P1,..., P_M)
    idx = 0
    N = int(input_data[idx])
    idx += 1
    M = int(input_data[idx])
    idx += 1
    
    # Read the colors of the sushi plates Takahashi ate
    plates = input_data[idx: idx + N]
    idx += N
    
    # Read the M colors that have special prices
    specials = input_data[idx: idx + M]
    idx += M
    
    # Read the prices: first price is P0 (default), then P1 ... PM corresponding to specials
    prices = list(map(int, input_data[idx: idx + M + 1]))
    # P0 is index 0, corresponding to colors that are not in specials
    idx += M + 1
    
    # Create a mapping from special color to its price
    special_price = {}
    for i in range(M):
        special_price[specials[i]] = prices[i + 1]
    
    total = 0
    # Calculate the total cost for the eaten sushi plates
    for color in plates:
        # If the color is in the special_price mapping, add its corresponding price
        # Otherwise, add the default price P0
        if color in special_price:
            total += special_price[color]
        else:
            total += prices[0]
    
    # Output the total cost
    sys.stdout.write(str(total))
    
if __name__ == '__main__':
    main()