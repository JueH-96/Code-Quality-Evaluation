# YOUR CODE HERE
n = int(input())
a = list(map(int, input().split()))

sequence = []

for i in range(n):
    # Add the i-th ball to the right end
    sequence.append(a[i])
    
    # Keep merging while possible
    while len(sequence) >= 2 and sequence[-1] == sequence[-2]:
        # Remove the last two balls and add a new one with size = sum of the two
        # Since sizes are 2^x, when we add 2^x + 2^x = 2^(x+1)
        last_size = sequence.pop()
        sequence.pop()
        sequence.append(last_size + 1)

print(len(sequence))