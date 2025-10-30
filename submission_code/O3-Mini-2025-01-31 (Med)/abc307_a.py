def main():
    import sys
    data = sys.stdin.read().split()
    N = int(data[0])
    steps = list(map(int, data[1:]))
    
    # Calculate sum for each week
    week_sums = []
    for i in range(N):
        week_sum = sum(steps[i*7:(i+1)*7])
        week_sums.append(week_sum)
    
    # Print the results separated by spaces
    print(" ".join(map(str, week_sums)))

if __name__ == "__main__":
    main()