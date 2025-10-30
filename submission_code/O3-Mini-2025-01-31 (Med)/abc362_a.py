def main():
    import sys
    input_data = sys.stdin.read().split()
    # The first three tokens are R, G, B (prices)
    R = int(input_data[0])
    G = int(input_data[1])
    B = int(input_data[2])
    # The fourth token is the color Takahashi dislikes
    disfavored_color = input_data[3]
    
    # Create a dictionary to map color name to cost
    costs = {
        "Red": R,
        "Green": G,
        "Blue": B
    }
    
    # Remove the cost of the disfavored pen
    del costs[disfavored_color]
    
    # The answer is the minimum cost among the remaining colors
    answer = min(costs.values())
    print(answer)

if __name__ == '__main__':
    main()