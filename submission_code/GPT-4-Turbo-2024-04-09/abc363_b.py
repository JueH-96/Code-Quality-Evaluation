def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    T = int(data[1])
    P = int(data[2])
    L = list(map(int, data[3:]))
    
    # Count how many people already have hair length >= T
    current_count = sum(1 for length in L if length >= T)
    
    # If the condition is already met, print 0
    if current_count >= P:
        print(0)
        return
    
    # Calculate the number of days required for each person to reach at least T
    days_needed = []
    for length in L:
        if length < T:
            days_needed.append(T - length)
    
    # Sort the days needed in ascending order
    days_needed.sort()
    
    # We need the P-th smallest number of days (0-based index P-1)
    # This will ensure that at least P people have hair length >= T
    if len(days_needed) >= P:
        print(days_needed[P-1])
    else:
        # If we don't have enough people to satisfy the condition, this should not happen
        # as P <= N and we only consider those who need more days to reach T.
        print(0)

if __name__ == "__main__":
    main()