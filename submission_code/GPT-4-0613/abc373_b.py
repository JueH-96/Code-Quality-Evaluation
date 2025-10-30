# YOUR CODE HERE
S = input().strip()
alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
positions = {c: i for i, c in enumerate(S)}
distances = [abs(positions[alphabet[i]] - positions[alphabet[i-1]]) for i in range(1, 26)]
print(sum(distances))