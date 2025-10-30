def generate_carpet(N):
    if N == 0:
        return ["#"]
    
    # Generate the smaller level-(N-1) carpet
    smaller_carpet = generate_carpet(N - 1)
    size = len(smaller_carpet)
    
    # Initialize the full carpet for level N
    full_carpet = []
    
    # Construct the 3x3 grid of level-(N-1) carpets
    for i in range(3):
        for row in smaller_carpet:
            if i == 1:
                # Middle row of blocks, center block should be white
                full_carpet.append(row + "." * size + row)
            else:
                # Top and bottom row of blocks, all are smaller carpets
                full_carpet.append(row * 3)
    
    return full_carpet

def main():
    import sys
    input = sys.stdin.read
    N = int(input().strip())
    
    # Generate the level-N carpet
    carpet = generate_carpet(N)
    
    # Print the carpet
    for line in carpet:
        print(line)