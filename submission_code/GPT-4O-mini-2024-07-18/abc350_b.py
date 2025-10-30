# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    Q = int(data[1])
    T = list(map(int, data[2:2 + Q]))
    
    # Initialize a set to track which holes have teeth
    teeth = set(range(1, N + 1))
    
    for t in T:
        if t in teeth:
            teeth.remove(t)  # Remove the tooth if it exists
        else:
            teeth.add(t)     # Add a tooth if it doesn't exist
    
    # The number of teeth left is the size of the set
    print(len(teeth))

if __name__ == "__main__":
    main()