def main():
    p, q = input().split()
    
    # Assign each letter to its position
    positions = {
        'A': 0,
        'B': 3,
        'C': 4,
        'D': 8,
        'E': 9,
        'F': 14,
        'G': 23
    }
    
    # Calculate and output the distance
    print(abs(positions[p] - positions[q]))

# Do not remove the call to main!
main()