def main():
    import sys
    input_data = sys.stdin.read().split()
    N = int(input_data[0])
    steps = list(map(int, input_data[1:]))
    
    week_sums = []
    for i in range(N):
        # Compute the sum for the i-th week (days: 7*i to 7*i+7)
        week_sum = sum(steps[7*i:7*i + 7])
        week_sums.append(week_sum)

    print(" ".join(map(str, week_sums)))

if __name__ == '__main__':
    main()