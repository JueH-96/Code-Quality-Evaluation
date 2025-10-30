def main():
    import sys
    data = sys.stdin.read().strip().split()
    N, T, P = map(int, data[:3])
    L = list(map(int, data[3:]))

    # For each person, calculate how many days they need 
    # to reach at least length T. If already >= T, need 0 days.
    days_needed = [(0 if length >= T else T - length) for length in L]
    days_needed.sort()

    # If there are already P or more people whose hair length is >= T 
    # (i.e. 0 days needed), print 0
    if days_needed.count(0) >= P:
        print(0)
        return

    # Otherwise, the answer is the (P-th smallest) day_needed value
    # which is days_needed[P-1] because of 0-based indexing.
    print(days_needed[P-1])

# Do not forget to call main()!
if __name__ == "__main__":
    main()