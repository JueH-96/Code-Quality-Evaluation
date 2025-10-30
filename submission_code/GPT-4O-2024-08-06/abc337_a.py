# YOUR CODE HERE
def determine_winner():
    import sys
    input = sys.stdin.read
    data = input().strip().split('
')
    
    N = int(data[0])
    total_takahashi = 0
    total_aoki = 0
    
    for i in range(1, N + 1):
        X, Y = map(int, data[i].split())
        total_takahashi += X
        total_aoki += Y
    
    if total_takahashi > total_aoki:
        print("Takahashi")
    elif total_aoki > total_takahashi:
        print("Aoki")
    else:
        print("Draw")

determine_winner()