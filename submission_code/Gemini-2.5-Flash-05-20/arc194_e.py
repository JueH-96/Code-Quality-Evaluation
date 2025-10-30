import sys

def solve():
    N, X, Y = map(int, sys.stdin.readline().split())
    S = sys.stdin.readline().strip()
    T = sys.stdin.readline().strip()

    s_val = [int(c) for c in S]
    t_val = [int(c) for c in T]

    current_balance = 0
    
    # Iterate through the string to calculate prefix balance
    # current_balance = (sum of 1s in S[0...k-1]) - (sum of 1s in T[0...k-1])
    for k in range(N):
        current_balance += (s_val[k] - t_val[k])
        
        # Check if the balance exceeds the allowed range [-X, Y]
        # An excess of '1's in S's prefix (current_balance > 0) can be at most Y.
        # An excess of '0's in S's prefix (current_balance < 0) can be at most X (so current_balance >= -X).
        if current_balance > Y or current_balance < -X:
            print("No")
            return

    # After iterating through the entire string, the total balance must be 0
    # This implies that the total count of '1's in S must be equal to the total count of '1's in T
    if current_balance != 0:
        print("No")
    else:
        print("Yes")

solve()