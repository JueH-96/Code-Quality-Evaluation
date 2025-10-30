# YOUR CODE HERE
N, Q = map(int, input().split())
treatments = list(map(int, input().split()))

teeth = [True] * N  # Initially all holes have teeth

for treatment in treatments:
    teeth[treatment - 1] = not teeth[treatment - 1]  # Toggle the state of the tooth

total_teeth = sum(teeth)
print(total_teeth)