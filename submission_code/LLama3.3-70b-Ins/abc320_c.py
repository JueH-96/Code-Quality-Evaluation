import sys

def find_min_time(M, S1, S2, S3):
    """
    Find the minimum possible number of seconds from the start of the spin until all the reels are stopped 
    so that all the displayed characters are the same.

    Args:
    M (int): The length of each reel.
    S1 (str): The arrangement of symbols on the first reel.
    S2 (str): The arrangement of symbols on the second reel.
    S3 (str): The arrangement of symbols on the third reel.

    Returns:
    int: The minimum possible number of seconds. If it's impossible, return -1.
    """
    min_time = float('inf')

    # Iterate over all possible characters
    for char in set(S1 + S2 + S3):
        # Check if the character exists in all reels
        if char not in S1 or char not in S2 or char not in S3:
            continue

        # Find the indices of the character in each reel
        idx1 = S1.index(char)
        idx2 = S2.index(char)
        idx3 = S3.index(char)

        # Calculate the time to stop each reel
        t1 = idx1
        t2 = idx2
        t3 = idx3

        # Find the maximum time
        max_time = max(t1, t2, t3)

        # Check if it's possible to stop all reels at the same time
        if (max_time - t1) % M == 0 and (max_time - t2) % M == 0 and (max_time - t3) % M == 0:
            min_time = min(min_time, max_time)

    # If no possible time is found, return -1
    if min_time == float('inf'):
        return -1
    else:
        return min_time

# Read inputs from stdin
M = int(sys.stdin.readline().strip())
S1 = sys.stdin.readline().strip()
S2 = sys.stdin.readline().strip()
S3 = sys.stdin.readline().strip()

# Find and print the minimum time
min_time = find_min_time(M, S1, S2, S3)
print(min_time)