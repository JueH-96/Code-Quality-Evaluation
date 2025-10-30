def solve():
    N = int(input())
    A = list(map(int, input().split()))
    
    sequence = []
    for a in A:
        # Add the i-th ball to the right end of the sequence
        sequence.append(a)
        
        # Apply the rules recursively
        while len(sequence) >= 2 and sequence[-1] == sequence[-2]:
            # Remove the rightmost two balls
            exponent = sequence.pop()
            sequence.pop()  # Remove the second rightmost ball
            # Add a new ball with an exponent increased by 1
            sequence.append(exponent + 1)
    
    return len(sequence)

print(solve())