# YOUR CODE HERE
N = int(input())

# Start checking from N
current = N
while True:
    # Extract digits
    hundreds = current // 100
    tens = (current // 10) % 10
    ones = current % 10
    
    # Check if it's a 326-like number
    if hundreds * tens == ones:
        print(current)
        break
    
    current += 1