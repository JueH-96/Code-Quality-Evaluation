# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    K = int(data[1])
    A = list(map(int, data[2:]))
    
    empty_seats = K
    attraction_starts = 0
    
    for group in A:
        if group > empty_seats:
            attraction_starts += 1
            empty_seats = K
        empty_seats -= group
    
    attraction_starts += 1  # For the last batch of people
    
    print(attraction_starts)

if __name__ == "__main__":
    main()