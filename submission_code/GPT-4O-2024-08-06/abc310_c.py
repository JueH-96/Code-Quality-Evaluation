# YOUR CODE HERE
def count_different_sticks():
    import sys
    input = sys.stdin.read
    data = input().splitlines()
    
    N = int(data[0])
    sticks = data[1:]
    
    # Use a set to store unique representations of sticks
    unique_sticks = set()
    
    for stick in sticks:
        # For each stick, consider both the stick and its reverse
        normal = stick
        reversed_stick = stick[::-1]
        
        # Add the lexicographically smaller one to the set
        if normal < reversed_stick:
            unique_sticks.add(normal)
        else:
            unique_sticks.add(reversed_stick)
    
    # The number of unique sticks is the size of the set
    print(len(unique_sticks))