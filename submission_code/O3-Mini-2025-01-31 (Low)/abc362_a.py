def main():
    import sys
    data = sys.stdin.read().split()
    R = int(data[0])
    G = int(data[1])
    B = int(data[2])
    C = data[3]
    
    # Dictionary mapping color to price.
    prices = {'Red': R, 'Green': G, 'Blue': B}
    
    # Remove the disliked pen color.
    prices.pop(C)
    
    # The minimum amount among the remaining options.
    answer = min(prices.values())
    print(answer)

if __name__ == '__main__':
    main()