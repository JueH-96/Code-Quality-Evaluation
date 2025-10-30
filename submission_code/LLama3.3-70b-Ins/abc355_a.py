import sys

def find_culprit(A, B):
    """
    Determine if the culprit can be uniquely identified based on the memories of the two witnesses.

    Args:
    A (int): The person that Ringo remembers is not the culprit.
    B (int): The person that Snuke remembers is not the culprit.

    Returns:
    int: The person's number if the culprit can be uniquely identified, otherwise -1.
    """
    # Create a set of all suspects
    suspects = {1, 2, 3}
    
    # Remove the person that Ringo remembers is not the culprit
    suspects.discard(A)
    
    # Remove the person that Snuke remembers is not the culprit
    suspects.discard(B)
    
    # If there is only one suspect left, return their number
    if len(suspects) == 1:
        return list(suspects)[0]
    # Otherwise, return -1
    else:
        return -1

def main():
    # Read the input from stdin
    A, B = map(int, sys.stdin.readline().split())
    
    # Find the culprit
    culprit = find_culprit(A, B)
    
    # Print the result
    print(culprit)

if __name__ == "__main__":
    main()