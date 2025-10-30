# YOUR CODE HERE
n = int(input())
a = list(map(int, input().split()))

sequence = []

for i in range(n):
    # Add the i-th ball to the sequence
    sequence.append(a[i])
    
    # Merge balls while possible
    while len(sequence) >= 2:
        # Check if the rightmost two balls have the same size
        if sequence[-1] == sequence[-2]:
            # Remove the two rightmost balls
            val = sequence.pop()
            sequence.pop()
            # Add a new ball with size 2^val + 2^val = 2^(val+1)
            sequence.append(val + 1)
        else:
            # Different sizes, stop merging
            break

print(len(sequence))