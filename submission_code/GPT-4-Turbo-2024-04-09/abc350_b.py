# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    Q = int(data[1])
    T = list(map(int, data[2:2+Q]))
    
    # Initially all holes have teeth
    teeth = [True] * N
    
    for t in T:
        index = t - 1  # Convert 1-based index to 0-based
        teeth[index] = not teeth[index]  # Toggle the tooth presence
    
    # Count how many teeth are present
    result = sum(teeth)
    
    print(result)

if __name__ == "__main__":
    main()