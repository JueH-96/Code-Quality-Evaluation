# YOUR CODE HERE
def solve():
    N, Q = map(int, input().split())
    H = [0] + list(map(int, input().split()))  # 1-indexed
    
    for _ in range(Q):
        l, r = map(int, input().split())
        count = 0
        
        # Check each building to the east of r
        for k in range(r + 1, N + 1):
            # Check if building k can be seen from both l and r
            # Need to check if max height in [l+1, k-1] < H[k]
            
            can_see = True
            max_height = 0
            
            # Find max height between l and k (exclusive)
            for i in range(l + 1, k):
                max_height = max(max_height, H[i])
            
            if max_height >= H[k]:
                can_see = False
            
            if can_see:
                count += 1
        
        print(count)

solve()