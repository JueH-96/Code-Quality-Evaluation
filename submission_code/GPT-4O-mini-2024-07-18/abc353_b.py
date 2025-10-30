def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    K = int(data[1])
    A = list(map(int, data[2:2 + N]))
    
    empty_seats = K
    attraction_starts = 0
    
    for group_size in A:
        if empty_seats < group_size:
            attraction_starts += 1
            empty_seats = K  # Reset empty seats
        empty_seats -= group_size  # Guide the group to the attraction
    
    # If there are still groups left after the last group, we need to start the attraction one last time
    if empty_seats < K:
        attraction_starts += 1
    
    print(attraction_starts)

if __name__ == "__main__":
    main()