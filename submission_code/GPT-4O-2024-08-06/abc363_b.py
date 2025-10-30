# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    T = int(data[1])
    P = int(data[2])
    L = list(map(int, data[3:]))
    
    # Calculate the number of people whose hair length is already >= T
    current_count = sum(1 for length in L if length >= T)
    
    # If the current count is already >= P, print 0
    if current_count >= P:
        print(0)
        return
    
    # Otherwise, calculate the number of days needed
    days = 0
    while current_count < P:
        days += 1
        current_count = sum(1 for length in L if length + days >= T)
    
    print(days)

if __name__ == "__main__":
    main()