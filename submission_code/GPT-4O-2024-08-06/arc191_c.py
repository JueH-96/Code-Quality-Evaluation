# YOUR CODE HERE
def solve():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    T = int(data[0])
    results = []
    
    for i in range(1, T + 1):
        N = int(data[i])
        A = 2
        M = (1 << N) - 1  # This is equivalent to 2^N - 1
        results.append(f"{A} {M}")
    
    print("
".join(results))