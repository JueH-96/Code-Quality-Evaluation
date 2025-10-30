def main():
    import sys
    data = sys.stdin.read().strip().split()
    if not data:
        return
    # Parse input values
    N = int(data[0])
    M = int(data[1])
    
    LIM = 10**9
    total = 0
    term = 1  # Represents N^i
    
    # Sum the geometric series from i=0 to M
    for i in range(M + 1):
        total += term
        
        # If at any point the running sum exceeds LIM, print "inf" and return
        if total > LIM:
            print("inf")
            return
        
        # If there are more terms to compute, prepare for the next term
        if i < M:
            # Check to avoid unnecessary multiplication if next term makes sum exceed LIM
            if term > LIM // N:  
                print("inf")
                return
            term *= N
    
    print(total)

if __name__ == '__main__':
    main()