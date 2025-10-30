def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    
    N = int(data[0])
    M = int(data[1])
    # Get the sequence A which has M distinct integers.
    A = set(map(int, data[2:2+M]))
    
    # Compute missing integers between 1 and N that are not in A.
    missing = [str(i) for i in range(1, N + 1) if i not in A]
    
    # Print the count of missing integers.
    print(len(missing))
    
    # If there are any missing numbers, print them space-separated.
    if missing:
        print(" ".join(missing))

if __name__ == '__main__':
    main()