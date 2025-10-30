def solve():
    import sys
    data = sys.stdin.read().split()
    N = int(data[0])
    A = list(map(int, data[1:]))

    operations = 0
    while True:
        # Count how many elements are positive
        positive_count = sum(x > 0 for x in A)
        
        # If the number of positive elements is less than 2, stop the process
        if positive_count < 2:
            break
        
        # Sort in descending order
        A.sort(reverse=True)
        
        # Decrease the top two elements by 1
        A[0] -= 1
        A[1] -= 1
        
        operations += 1
    
    print(operations)

def main():
    solve()

if __name__ == "__main__":
    main()