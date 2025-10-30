def main():
    # Read the number of people
    n = int(input())
    
    # Read the final scores of persons 1 to N-1
    scores = list(map(int, input().split()))
    
    # Calculate the final score of person N
    # Since the sum of all scores must be 0,
    # the score of person N must be the negative sum of all other scores
    result = -sum(scores)
    
    # Print the result
    print(result)

if __name__ == "__main__":
    main()