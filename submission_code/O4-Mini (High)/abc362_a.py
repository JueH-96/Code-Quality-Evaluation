def main():
    # Read the prices for red, green, and blue pens
    R, G, B = map(int, input().split())
    # Read the disliked color
    C = input().strip()
    
    # Depending on the disliked color, compute the minimum cost among the other two
    if C == "Red":
        answer = min(G, B)
    elif C == "Green":
        answer = min(R, B)
    else:  # C == "Blue"
        answer = min(R, G)
    
    # Print the result
    print(answer)

# Call the main function
main()