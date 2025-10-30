import sys

def read_input():
    """Read input from stdin."""
    lines = sys.stdin.readlines()
    N, M = map(int, lines[0].split())
    photos = [list(map(int, line.split())) for line in lines[1:]]
    return N, M, photos

def find_bad_mood_pairs(N, M, photos):
    """Find pairs of people who may be in a bad mood."""
    # Create a set to store pairs of people who stood next to each other
    stood_next_to = set()
    
    # Iterate over each photo
    for photo in photos:
        # Iterate over each person in the photo
        for i in range(N - 1):
            # Add the pair of people who stood next to each other to the set
            stood_next_to.add(tuple(sorted((photo[i], photo[i + 1]))))
    
    # Initialize the count of bad mood pairs
    bad_mood_pairs = 0
    
    # Iterate over each pair of people
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            # If the pair did not stand next to each other in any photo, increment the count
            if tuple(sorted((i, j))) not in stood_next_to:
                bad_mood_pairs += 1
    
    return bad_mood_pairs

def main():
    """Main function."""
    N, M, photos = read_input()
    bad_mood_pairs = find_bad_mood_pairs(N, M, photos)
    print(bad_mood_pairs)

if __name__ == "__main__":
    main()