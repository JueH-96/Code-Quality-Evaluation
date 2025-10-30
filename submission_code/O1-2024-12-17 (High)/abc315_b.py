def main():
    import sys
    data = sys.stdin.read().strip().split()
    M = int(data[0])
    D = list(map(int, data[1:] ))
    
    # Sum of all days in the year
    total_days = sum(D)
    
    # The target day index (middle day of the year)
    mid = (total_days + 1) // 2
    
    # Find which month/day this corresponds to
    cumulative = 0
    for i in range(M):
        if cumulative + D[i] >= mid:
            day_of_month = mid - cumulative
            print(i+1, day_of_month)
            return
        cumulative += D[i]

# Do not forget to actually call main() to run the solution.
if __name__ == '__main__':
    main()