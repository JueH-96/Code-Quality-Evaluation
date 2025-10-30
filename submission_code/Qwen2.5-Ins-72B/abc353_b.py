# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    K = int(data[1])
    A = list(map(int, data[2:]))
    
    empty_seats = K
    start_count = 0
    
    for group in A:
        if empty_seats < group:
            start_count += 1
            empty_seats = K
        empty_seats -= group
    
    if empty_seats < K:
        start_count += 1
    
    print(start_count)

if __name__ == "__main__":
    main()