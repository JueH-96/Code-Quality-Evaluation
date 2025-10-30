# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    T = int(data[1])
    P = int(data[2])
    L = list(map(int, data[3:]))
    
    # Check if the condition is already satisfied
    count = sum(1 for length in L if length >= T)
    if count >= P:
        print(0)
        return
    
    # Calculate the number of days needed
    days = 0
    while count < P:
        days += 1
        count = sum(1 for length in L if length + days >= T)
    
    print(days)

if __name__ == "__main__":
    main()