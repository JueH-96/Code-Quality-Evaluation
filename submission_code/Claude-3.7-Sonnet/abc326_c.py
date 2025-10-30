def main():
    N, M = map(int, input().split())
    A = list(map(int, input().split()))
    
    # Sort the gifts by their coordinates
    A.sort()
    
    max_gifts = 0
    right = 0
    
    for left in range(N):
        # For each left pointer, consider the interval [A[left], A[left] + M)
        # and include all gifts within this interval
        while right < N and A[right] < A[left] + M:
            right += 1
        
        # right - left is the number of gifts in the current interval
        max_gifts = max(max_gifts, right - left)
    
    print(max_gifts)

if __name__ == "__main__":
    main()