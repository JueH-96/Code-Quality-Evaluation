# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    D = int(data[1])
    S = data[2]
    
    # Count the number of empty boxes initially
    initial_empty_boxes = S.count('.')
    
    # After D days, D more boxes will be empty
    total_empty_boxes = initial_empty_boxes + D
    
    # Print the result
    print(total_empty_boxes)