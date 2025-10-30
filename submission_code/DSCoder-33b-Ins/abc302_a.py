# YOUR CODE HERE
import sys

A, B = map(int, sys.stdin.readline().split())

# Calculate the number of attacks needed to reduce the enemy's stamina to 0 or less
attacks = (A + B - 1) // B

print(attacks)