# YOUR CODE HERE
def count_arithmetic_sequence_integers(A, B):
    if A == B:
        return 1
    
    count = 0
    for x in range(-100, 201):
        if (A + B) % 2 == 0 and x == (A + B) // 2:
            count += 1
        elif x == 2 * A - B or x == 2 * B - A:
            count += 1
    
    return count

import sys
input = sys.stdin.read
A, B = map(int, input().strip().split())
print(count_arithmetic_sequence_integers(A, B))