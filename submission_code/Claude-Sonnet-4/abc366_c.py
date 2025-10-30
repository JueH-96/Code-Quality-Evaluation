# YOUR CODE HERE
Q = int(input())
bag_count = {}  # Dictionary to count occurrences of each integer
distinct_count = 0  # Number of distinct integers in the bag

for _ in range(Q):
    query = input().split()
    
    if query[0] == '1':
        # Add ball with integer x
        x = int(query[1])
        if x in bag_count:
            bag_count[x] += 1
        else:
            bag_count[x] = 1
            distinct_count += 1
    
    elif query[0] == '2':
        # Remove ball with integer x
        x = int(query[1])
        bag_count[x] -= 1
        if bag_count[x] == 0:
            del bag_count[x]
            distinct_count -= 1
    
    else:  # query[0] == '3'
        # Print number of different integers
        print(distinct_count)