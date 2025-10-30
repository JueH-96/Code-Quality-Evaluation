# YOUR CODE HERE
def determine_winner():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    A = list(map(int, data[1:]))
    
    total_moves = sum(A)
    
    # Determine the winner based on the parity of total_moves - N
    if (total_moves - N) % 2 == 0:
        print("Snuke")
    else:
        print("Fennec")

determine_winner()