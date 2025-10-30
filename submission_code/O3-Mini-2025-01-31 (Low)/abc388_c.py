def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    n = int(data[0])
    A = list(map(int, data[1:]))
    
    count = 0
    i = 0  # pointer for potential top mochi index
    
    # For each candidate bottom (index j), count how many indices < j satisfy A[i] <= A[j]/2.
    for j in range(n):
        # Move pointer i forward while it is a valid top candidate
        while i < j and A[i] <= A[j] / 2:
            i += 1
        # All indices < i satisfy the condition.
        count += i
    sys.stdout.write(str(count))
    
if __name__ == '__main__':
    main()