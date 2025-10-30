def max_experience(N, A):
    total_experience = 0
    defeated_count = 0

    for strength in A:
        # If we defeat the monster, we gain experience
        total_experience += strength
        defeated_count += 1
        
        # If it's an even-numbered defeated monster, we gain additional experience
        if defeated_count % 2 == 0:
            total_experience += strength

    return total_experience

# Read input
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
A = list(map(int, data[1:N+1]))

# Calculate and print the result
result = max_experience(N, A)
print(result)