from collections import defaultdict

def main():
    N = int(input())
    A_str = input().split()
    A = [int(x) for x in A_str]

    # Using defaultdict to store 0-indexed positions of each color.
    # color_positions[color] will be a list of indices where 'color' appears.
    color_positions = defaultdict(list)
    
    # Populate color_positions
    # Iterate through the array A with their 0-indexed positions
    for i in range(2 * N):
        color = A[i]
        color_positions[color].append(i) # Append the current index

    ans_count = 0 # Initialize count of colors satisfying the condition

    # Iterate through each color from 1 to N
    for color_val in range(1, N + 1):
        # Get the list of 0-indexed positions for the current color_val.
        # Each color appears exactly twice, so this list will contain two indices.
        positions = color_positions[color_val]
        
        # positions[0] is the index of the first occurrence (smaller index)
        # positions[1] is the index of the second occurrence (larger index)
        # This is guaranteed because we iterated through A from left to right.
        idx1 = positions[0]
        idx2 = positions[1]

        # The condition "exactly one person between the two people" means
        # their 0-indexed positions are of the form k and k+2.
        # For example, if indices are 0 and 2, person at index 1 is between them.
        # The difference between such indices is (k+2) - k = 2.
        if idx2 - idx1 == 2:
            ans_count += 1
    
    print(ans_count)

if __name__ == '__main__':
    main()