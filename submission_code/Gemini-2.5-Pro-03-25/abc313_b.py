# YOUR CODE HERE
import sys

def solve():
    """
    Solves the problem of finding the uniquely determined strongest programmer
    based on partial information about relative strengths.
    Reads input from stdin, processes the information, and prints the result to stdout.
    """
    
    # Read N (number of programmers) and M (number of information pieces) from the first line of input.
    # N programmers are numbered 1 to N.
    N, M = map(int, sys.stdin.readline().split())
    
    # Initialize a set containing all programmer IDs from 1 to N.
    # These programmers are initially considered potential candidates for being the strongest.
    potentially_strongest = set(range(1, N + 1))
    
    # Process each of the M pieces of information provided.
    for _ in range(M):
        # Read a pair (A, B) from input. This represents the information that "person A is stronger than person B".
        A, B = map(int, sys.stdin.readline().split())
        
        # The information "A is stronger than B" implies that B cannot possibly be the strongest programmer,
        # because there exists at least one person (A) who is stronger than B.
        # Therefore, we remove B from the set of potential candidates.
        # We use the `discard` method which safely removes an element if it exists in the set,
        # and does nothing if the element is not present (e.g., if B was already removed based on prior information).
        potentially_strongest.discard(B) 
            
    # After processing all M pieces of information, examine the size of the set `potentially_strongest`.
    if len(potentially_strongest) == 1:
        # If exactly one candidate remains in the set, this means based on the given information,
        # only this one person could possibly be the strongest.
        # The problem guarantees that a unique strongest person exists in the complete ranking.
        # If our partial information narrows it down to one possibility, that must be the one.
        # We can uniquely determine the strongest programmer.
        
        # Extract the single element from the set. Converting the set to a list and taking the first element is one way.
        strongest_candidate = list(potentially_strongest)[0]
        
        # Print the number of the uniquely determined strongest programmer.
        print(strongest_candidate)
    else:
        # If the number of candidates remaining in the set is not equal to 1:
        # Case 1: len(potentially_strongest) == 0. This case should theoretically not happen based on the problem constraints
        # which guarantee the existence of a strongest person and consistency of information. If it did, it would mean
        # everyone is shown to be weaker than someone, which contradicts the existence of a strongest person.
        # Case 2: len(potentially_strongest) > 1. This means there are multiple programmers who have not been shown to be weaker
        # than anyone else based on the given partial information. Any of these remaining candidates could potentially be the
        # strongest, depending on the unknown strength relationships between them and possibly others.
        # In such cases, we cannot uniquely determine the strongest programmer from the given information.
        
        # As required by the problem statement, print -1 when the strongest programmer cannot be uniquely determined.
        print("-1")

# Call the solve function to execute the program logic.
solve()