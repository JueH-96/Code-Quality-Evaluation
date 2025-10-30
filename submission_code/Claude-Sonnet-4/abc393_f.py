import bisect

def solve():
    N, Q = map(int, input().split())
    A = list(map(int, input().split()))
    
    for _ in range(Q):
        R, X = map(int, input().split())
        
        # Consider only first R elements and filter by X
        valid_elements = []
        for i in range(R):
            if A[i] <= X:
                valid_elements.append(A[i])
        
        # Find LIS of valid_elements
        if not valid_elements:
            print(0)
            continue
            
        # Use patience sorting approach for LIS
        tails = []
        
        for num in valid_elements:
            # Find the position where num should be inserted
            pos = bisect.bisect_left(tails, num)
            
            if pos == len(tails):
                tails.append(num)
            else:
                tails[pos] = num
        
        print(len(tails))

solve()