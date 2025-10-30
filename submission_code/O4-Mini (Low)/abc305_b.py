def main():
    import sys
    data = sys.stdin.read().strip().split()
    p, q = data[0], data[1]
    
    # Precomputed positions of each point on the line (distance from A)
    positions = {
        'A': 0,
        'B': 3,
        'C': 4,
        'D': 8,
        'E': 9,
        'F': 14,
        'G': 23
    }
    
    # Compute and print the absolute difference
    distance = abs(positions[p] - positions[q])
    print(distance)

if __name__ == "__main__":
    main()