def main():
    import sys

    p, q = sys.stdin.readline().split()
    
    # Precomputed positions (distance from A)
    positions = {
        'A': 0,
        'B': 3,
        'C': 4,
        'D': 8,
        'E': 9,
        'F': 14,
        'G': 23
    }
    
    # Calculate the absolute difference
    distance = abs(positions[p] - positions[q])
    print(distance)

# Do not remove the function call below
main()