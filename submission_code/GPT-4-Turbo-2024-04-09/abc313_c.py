def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    A = list(map(int, data[1:]))
    
    if N == 1:
        print(0)
        return
    
    # Calculate the total sum of the array
    total_sum = sum(A)
    
    # Calculate the ideal average value for minimal difference
    ideal_lower = total_sum // N
    ideal_upper = ideal_lower + 1
    
    # Calculate how many numbers should be ideal_lower and ideal_upper
    count_upper = total_sum % N
    count_lower = N - count_upper
    
    # Calculate the current counts of ideal_lower and ideal_upper
    current_lower = sum(1 for x in A if x == ideal_lower)
    current_upper = sum(1 for x in A if x == ideal_upper)
    
    # Calculate the excess and deficit
    excess_lower = sum(x - ideal_lower for x in A if x < ideal_lower)
    excess_upper = sum(x - ideal_upper for x in A if x > ideal_upper)
    
    # The number of operations needed is the maximum of the excesses
    # because we need to balance the excess by moving it to the deficit areas
    print(max(excess_lower, excess_upper))

if __name__ == "__main__":
    main()