# YOUR CODE HERE
def solve():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    index = 0
    T = int(data[index])
    index += 1
    results = []
    
    for _ in range(T):
        N = int(data[index])
        index += 1
        A = list(map(int, data[index:index + 2 * N]))
        index += 2 * N
        
        # Dictionary to store positions of each number
        positions = {}
        
        for i in range(2 * N):
            num = A[i]
            if num not in positions:
                positions[num] = []
            positions[num].append(i)
        
        # Count valid pairs
        count = 0
        
        # Check all pairs (a, b) with 1 <= a < b <= N
        for a in range(1, N + 1):
            for b in range(a + 1, N + 1):
                pos_a = positions[a]
                pos_b = positions[b]
                
                # Check if both a and b are not adjacent
                if abs(pos_a[0] - pos_a[1]) == 1 or abs(pos_b[0] - pos_b[1]) == 1:
                    continue
                
                # Check if we can make both adjacent by swapping
                # We need to ensure that the positions do not overlap in a way that prevents adjacency
                if (pos_a[0] < pos_b[0] < pos_a[1] < pos_b[1]) or (pos_b[0] < pos_a[0] < pos_b[1] < pos_a[1]):
                    count += 1
        
        results.append(count)
    
    for result in results:
        print(result)