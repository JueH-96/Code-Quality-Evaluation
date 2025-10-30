def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    N = int(data[0])
    T = int(data[1])
    P = int(data[2])
    hair_lengths = list(map(int, data[3:]))

    # Count the number of people whose hair is already at least T
    current_count = sum(1 for length in hair_lengths if length >= T)
    
    # If the current count meets or exceeds P, print 0 days.
    if current_count >= P:
        print(0)
        return
    
    # For each person with hair length less than T, compute the number of days
    # required to reach length T. Then sort these required days.
    required_days = sorted(T - length for length in hair_lengths if length < T)
    
    # We need (P - current_count) more people to reach length T.
    # Since the list is sorted in ascending order, the smallest number of days needed
    # to get at least that many people is given by the (P - current_count - 1)th element.
    answer = required_days[(P - current_count) - 1]
    
    print(answer)

if __name__ == '__main__':
    main()