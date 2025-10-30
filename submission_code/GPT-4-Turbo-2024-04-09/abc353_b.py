def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    K = int(data[1])
    A = list(map(int, data[2:]))
    
    current_seats = K
    starts = 0
    
    for people in A:
        if people > current_seats:
            starts += 1
            current_seats = K
        
        current_seats -= people
    
    starts += 1  # For the final start after the last group is seated
    
    print(starts)

if __name__ == "__main__":
    main()