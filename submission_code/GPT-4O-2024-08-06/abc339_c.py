# YOUR CODE HERE
def minimum_passengers():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    A = list(map(int, data[1:]))
    
    current_sum = 0
    min_prefix = 0
    
    for change in A:
        current_sum += change
        if current_sum < min_prefix:
            min_prefix = current_sum
    
    # The minimum initial passengers needed to never go negative
    print(-min_prefix)