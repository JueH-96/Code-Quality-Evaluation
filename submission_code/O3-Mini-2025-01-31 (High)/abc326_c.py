def main():
    import sys

    data = sys.stdin.read().split()
    if not data:
        return
    N = int(data[0])
    M = int(data[1])
    A = list(map(int, data[2:2+N]))
    
    # Sort the gift positions to apply the two-pointer sliding window technique.
    A.sort()
    
    ans = 0
    j = 0
    # Use i as the left pointer of our sliding window.
    for i in range(N):
        # Move the right pointer j forward as long as A[j] is within the interval [A[i], A[i]+M)
        while j < N and A[j] < A[i] + M:
            j += 1
        # The number of gifts in the current window is j - i
        ans = max(ans, j - i)
        
    sys.stdout.write(str(ans))
    
if __name__ == "__main__":
    main()