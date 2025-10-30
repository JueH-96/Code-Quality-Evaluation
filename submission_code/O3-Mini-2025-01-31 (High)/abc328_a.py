def main():
    import sys
    # Read all input data and split it into tokens
    data = sys.stdin.read().split()
    
    # First two tokens are N (number of problems) and X (the score threshold)
    N = int(data[0])
    X = int(data[1])
    
    # The next N tokens are the scores for each problem
    scores = list(map(int, data[2:2+N]))
    
    # Calculate the sum of scores that are X or less
    result = sum(score for score in scores if score <= X)
    
    # Print the result
    print(result)

# Call the main function
if __name__ == "__main__":
    main()