# YOUR CODE HERE
def count_takahashi():
    import sys
    input = sys.stdin.read
    data = input().splitlines()
    
    N = int(data[0])
    count = 0
    
    for i in range(1, N + 1):
        if data[i] == "Takahashi":
            count += 1
    
    print(count)