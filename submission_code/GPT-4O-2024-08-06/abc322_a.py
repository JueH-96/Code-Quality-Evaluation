# YOUR CODE HERE
def find_abc_position():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    S = data[1]
    
    # Iterate through the string and check for "ABC"
    for i in range(N - 2):
        if S[i:i+3] == "ABC":
            print(i + 1)
            return
    
    # If "ABC" is not found, print -1
    print(-1)