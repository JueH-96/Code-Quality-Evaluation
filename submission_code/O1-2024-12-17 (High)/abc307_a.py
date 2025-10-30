# YOUR CODE HERE
def main():
    import sys
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    steps = list(map(int, data[1:]))

    # Calculate weekly sums
    results = []
    for i in range(N):
        weekly_sum = sum(steps[7*i:7*i+7])
        results.append(str(weekly_sum))

    # Output the results
    print(" ".join(results))

if __name__ == "__main__":
    main()