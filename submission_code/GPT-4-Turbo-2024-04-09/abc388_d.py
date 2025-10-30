import sys
input = sys.stdin.read

def solve():
    data = input().split()
    N = int(data[0])
    A = list(map(int, data[1:]))
    
    # This will hold the number of stones each alien has after N years
    B = [0] * N
    
    # This will track the number of stones each alien has at any given time
    C = A[:]
    
    # This will track the number of adults who can give stones
    adults_with_stones = 0
    
    for i in range(N):
        # The current alien becomes an adult
        # They receive stones from all current adults with stones
        C[i] += adults_with_stones
        
        # Now, decrease the stone count of all adults who had stones
        if C[i] > 0:
            adults_with_stones += 1
        
        # Set the final count of stones for this alien
        B[i] = C[i]
        
        # After setting, this alien will give a stone to the next adult if they have any
        if C[i] > 0:
            C[i] -= 1
    
    print(" ".join(map(str, B)))