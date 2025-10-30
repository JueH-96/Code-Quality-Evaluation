def main():
    import sys
    # Read the entire input
    input_data = sys.stdin.read().strip().split()
    # The first element is N
    N = int(input_data[0])
    # Next elements are the steps, convert them to integers
    steps = list(map(int, input_data[1:]))
    
    results = []
    # Process each week by summing each group of 7 days
    for i in range(N):
        weekly_sum = sum(steps[7*i : 7*i + 7])
        results.append(str(weekly_sum))
    
    # Print the results separated by spaces
    print(" ".join(results))

# Call the main function to execute the solution
if __name__ == '__main__':
    main()