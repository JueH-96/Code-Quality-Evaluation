def count_different_sticks(n, sticks):
    unique_sticks = set()
    
    for stick in sticks:
        # Add both the stick and its reverse to the set
        unique_sticks.add(stick)
        unique_sticks.add(stick[::-1])
    
    # The number of unique sticks is the size of the set
    return len(unique_sticks)

# Read input
import sys
input = sys.stdin.read
data = input().splitlines()

N = int(data[0])
S = data[1:N+1]

# Get the result and print it
result = count_different_sticks(N, S)
print(result)