# YOUR CODE HERE

import sys

# Initialize an empty set to store the unique integers
unique_integers = set()

# Read the number of queries
Q = int(sys.stdin.readline().strip())

for _ in range(Q):
    query = sys.stdin.readline().strip().split()
    
    if len(query) == 1:
        # If the query is of type 1, put the integer x on the bag
        x = int(query[0])
        unique_integers.add(x)
    elif len(query) == 2:
        # If the query is of type 2, remove the integer x from the bag
        x = int(query[1])
        if x in unique_integers:
            unique_integers.remove(x)
    elif len(query) == 1:
        # If the query is of type 3, print the number of different integers written on the balls in the bag
        print(len(unique_integers))