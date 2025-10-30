def main():
    # Read N (number of problems) and X (threshold score)
    N, X = map(int, input().split())
    # Read the list of scores
    scores = list(map(int, input().split()))
    # Sum up all scores that are <= X
    total = sum(s for s in scores if s <= X)
    # Print the result
    print(total)

# Call the main function
main()