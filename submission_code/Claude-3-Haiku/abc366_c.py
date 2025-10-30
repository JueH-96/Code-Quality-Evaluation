from collections import set

# Read the number of queries
Q = int(input())

# Initialize an empty set to store the unique integers
unique_integers = set()

for _ in range(Q):
    query = input().split()
    query_type = int(query[0])
    
    if query_type == 1:
        # Add the integer to the set
        unique_integers.add(int(query[1]))
    elif query_type == 2:
        # Remove the integer from the set
        unique_integers.remove(int(query[1]))
    else:
        # Print the number of unique integers
        print(len(unique_integers))