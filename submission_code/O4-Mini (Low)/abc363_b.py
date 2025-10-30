def main():
    import sys
    data = sys.stdin.read().split()
    N = int(data[0])
    T = int(data[1])
    P = int(data[2])
    L = list(map(int, data[3:]))

    # For each person, compute how many days it takes for their hair to reach at least T.
    # If already >= T, that's 0 days.
    days_needed = [max(0, T - length) for length in L]
    
    # Sort these required days in ascending order.
    days_needed.sort()
    
    # The P-th smallest number of days is the moment when at least P people have hair >= T.
    # Since list is 0-indexed, we take index P-1.
    answer = days_needed[P-1]
    
    print(answer)

if __name__ == "__main__":
    main()