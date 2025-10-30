def main():
    import sys
    input_line = sys.stdin.read().strip()
    if input_line:
        N, X, Y, Z = map(int, input_line.split())
        
        # Determine the direction of travel
        if X < Y:
            # Inbound train: stations in increasing order from X to Y.
            route = range(X, Y + 1)
        else:
            # Outbound train: stations in decreasing order from X to Y.
            route = range(X, Y - 1, -1)
        
        # Check if station Z is in the route.
        if Z in route:
            print("Yes")
        else:
            print("No")
            
if __name__ == "__main__":
    main()