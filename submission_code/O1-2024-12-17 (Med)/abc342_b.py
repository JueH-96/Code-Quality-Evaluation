def main():
    import sys
    data = sys.stdin.read().strip().split()
    
    N = int(data[0])
    P = list(map(int, data[1:N+1]))
    Q = int(data[N+1])
    
    # Create a dictionary to get the index (position) of each person quickly
    position_map = {}
    for i, person in enumerate(P):
        position_map[person] = i  # 0-based index
    
    idx = N + 2
    for _ in range(Q):
        A = int(data[idx])
        B = int(data[idx+1])
        idx += 2
        
        # Compare the positions of A and B in the lineup
        if position_map[A] < position_map[B]:
            print(A)
        else:
            print(B)

# Call main
main()