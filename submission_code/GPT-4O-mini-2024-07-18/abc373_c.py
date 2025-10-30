# YOUR CODE HERE
def max_sum_of_sequences():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    A = list(map(int, data[1:N+1]))
    B = list(map(int, data[N+1:2*N+1]))
    
    max_A = max(A)
    max_B = max(B)
    
    result = max_A + max_B
    print(result)

max_sum_of_sequences()