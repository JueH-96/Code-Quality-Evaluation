def main():
    import sys
    
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    A = list(map(int, data[1:]))

    count = 0
    while True:
        # Count the number of positive elements
        positive_count = sum(1 for x in A if x > 0)
        if positive_count < 2:
            break
        
        # Sort in descending order
        A.sort(reverse=True)
        
        # Decrease the top two elements
        A[0] -= 1
        A[1] -= 1
        
        count += 1
    
    print(count)

if __name__ == "__main__":
    main()