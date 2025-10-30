def main():
    import sys
    
    data = sys.stdin.read().strip().split()
    if not data:
        return
    N = int(data[0])
    A = list(map(int, data[1:]))
    
    count = 0
    while True:
        # Check if there are at least two positive integers
        positives = [x for x in A if x > 0]
        if len(positives) <= 1:
            break
        
        # Sort descending
        A.sort(reverse=True)
        # Decrease the two largest elements by 1 if they are positive
        if A[0] > 0:
            A[0] -= 1
        if A[1] > 0:
            A[1] -= 1

        count += 1
        
    sys.stdout.write(str(count))
    
if __name__ == '__main__':
    main()