# YOUR CODE HERE
import sys

def solve():
    # Read N and K from the first line
    N, K = map(int, sys.stdin.readline().split())
    # Read the tooth string S from the second line
    S = sys.stdin.readline().strip()

    # Convert the string S to a list of characters for mutability
    # This allows us to change 'O' to 'X' in-place
    s_list = list(S)
    
    strawberry_count = 0
    # Initialize the scanning index
    i = 0

    # Iterate through the teeth. The loop continues as long as
    # there's a possibility to form a block of K teeth starting at index i.
    # The last possible starting index for a K-block is N-K.
    while i <= N - K:
        # Check if the current K teeth from index i are all 'O' (healthy)
        # We can use all() with a generator expression for concise checking
        if all(s_list[i + j] == 'O' for j in range(K)):
            # If all K teeth are healthy, we can eat a strawberry
            strawberry_count += 1
            
            # Mark these K teeth as 'X' (cavities)
            for j in range(K):
                s_list[i + j] = 'X'
            
            # Advance the scanning index by K because these K teeth are now used up
            # and cannot be part of any future strawberry-eating operation.
            i += K
        else:
            # If the current K teeth are not all healthy,
            # move to the next tooth to continue scanning.
            i += 1
            
    # Print the maximum number of strawberries
    print(strawberry_count)

# Call the solve function to execute the program
solve()