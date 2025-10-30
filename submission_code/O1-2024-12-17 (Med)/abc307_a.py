def main():
    import sys

    data = list(map(int, sys.stdin.read().strip().split()))
    N = data[0]
    steps = data[1:]  # The 7*N step counts
    
    week_sums = []
    for i in range(N):
        start_index = i * 7
        end_index = start_index + 7
        week_sum = sum(steps[start_index:end_index])
        week_sums.append(week_sum)
    
    print(" ".join(map(str, week_sums)))

# Do not remove this call
if __name__ == "__main__":
    main()