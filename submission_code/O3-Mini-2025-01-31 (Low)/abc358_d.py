def main():
    import sys
    input = sys.stdin.readline
    
    # read input
    N, M = map(int, input().split())
    A = list(map(int, input().split()))
    B = list(map(int, input().split()))
    
    # sort the boxes and demands in non-decreasing order
    A.sort()
    B.sort()
    
    total_cost = 0
    j = 0  # pointer for boxes A
    
    # process each demand in ascending order
    for b in B:
        # find the smallest available box that has at least b candies
        while j < N and A[j] < b:
            j += 1
            
        # if we run out of boxes that can satisfy the demand, print -1
        if j == N:
            print(-1)
            return
        
        # add the cost (which is equal to the number of candies) of the chosen box
        total_cost += A[j]
        
        # move to the next box (we cannot use the same box again)
        j += 1
        
    # output the minimum cost
    print(total_cost)

if __name__ == '__main__':
    main()