# YOUR CODE HERE
n = int(input())
a = list(map(int, input().split()))

# Create a list to store the order of people
order = [0] * n

# Find the person at the front of the line
front = a.index(-1)
order[0] = front + 1  # +1 because people are numbered from 1

# Fill the rest of the order
for i in range(1, n):
    # Find the person who is in front of the current person
    for j in range(n):
        if a[j] == order[i-1]:
            order[i] = j + 1  # +1 because people are numbered from 1
            break

# Print the order
print(' '.join(map(str, order)))