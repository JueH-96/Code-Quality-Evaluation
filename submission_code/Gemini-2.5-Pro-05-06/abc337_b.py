# YOUR CODE HERE
def solve():
    S = input()
    N = len(S)
    
    ptr = 0
    
    # Scan for A's (S_A part)
    while ptr < N and S[ptr] == 'A':
        ptr += 1
        
    # Scan for B's (S_B part)
    while ptr < N and S[ptr] == 'B':
        ptr += 1
        
    # Scan for C's (S_C part)
    while ptr < N and S[ptr] == 'C':
        ptr += 1
        
    if ptr == N:
        print("Yes")
    else:
        print("No")

if __name__ == '__main__':
    solve()