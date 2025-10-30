# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    Q = int(data[1])
    treatments = list(map(int, data[2:]))
    
    # Initialize all holes with teeth
    teeth = [True] * N
    
    # Perform each treatment
    for t in treatments:
        index = t - 1
        # Toggle the presence of a tooth in the hole
        teeth[index] = not teeth[index]
    
    # Count the number of teeth remaining
    remaining_teeth = sum(teeth)
    
    # Output the result
    print(remaining_teeth)

if __name__ == "__main__":
    main()