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
        A = data[index:index + N]
        index += N
        
        # Count the number of contiguous blocks
        blocks = 1
        for i in range(1, N):
            if A[i] != A[i - 1]:
                blocks += 1
        
        results.append(blocks)
    
    # Print all results for each test case
    sys.stdout.write('
'.join(map(str, results)) + '
')