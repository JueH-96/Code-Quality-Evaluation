import sys

def main():
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    chords = [(int(data[i]), int(data[i+1])) for i in range(1, 2*N, 2)]
    
    # Normalize the points to be between 0 and 2*N-1
    chords = [(a-1, b-1) for a, b in chords]
    
    # Sort the chords based on the starting point
    chords.sort()
    
    # Check for intersections
    max_end = -1
    for start, end in chords:
        if start > end:
            end += 2 * N
        if start < max_end:
            print("Yes")
            return
        max_end = max(max_end, end)
    
    print("No")

if __name__ == "__main__":
    main()