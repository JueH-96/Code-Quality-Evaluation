def main():
    import sys
    input = sys.stdin.readline
    
    N = int(input().strip())
    A = list(map(int, input().split()))
    
    # Build a mapping from a person to the person who stands immediately behind them.
    # Since there's exactly one arrangement, each person (except the front)
    # appears exactly once as a follower.
    follower = [None] * (N + 1)  
    front = None
    
    for i in range(1, N + 1):
        if A[i - 1] == -1:
            front = i
        else:
            # Person i is behind person A[i-1]
            follower[A[i - 1]] = i
    
    # Reconstruct the line starting from the front.
    res = []
    current = front
    while current is not None:
        res.append(str(current))
        # Go to the person behind the current person.
        if follower[current] is not None:
            current = follower[current]
        else:
            current = None
    
    print(" ".join(res))

if __name__ == '__main__':
    main()