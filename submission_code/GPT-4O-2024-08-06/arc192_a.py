# YOUR CODE HERE
def is_good_string_possible(n, a):
    # If there are no zeros, any string is good
    if all(x == 1 for x in a):
        return "Yes"
    
    # Check for any consecutive zeros
    for i in range(n):
        if a[i] == 0 and a[(i + 1) % n] == 0:
            return "Yes"
    
    # If no such pair is found, return No
    return "No"

import sys
input = sys.stdin.read
data = input().split()
n = int(data[0])
a = list(map(int, data[1:]))

print(is_good_string_possible(n, a))