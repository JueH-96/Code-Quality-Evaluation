import sys

# Reading input from stdin
input_data = sys.stdin.read().split()
N, Q = map(int, input_data[:2])
treatments = list(map(int, input_data[2:]))

# Initializing the state of the holes (1 for tooth, 0 for no tooth)
holes = [1] * N

# Performing the treatments
for treatment in treatments:
    hole_index = treatment - 1  # Adjusting for 0-based indexing
    holes[hole_index] = 1 - holes[hole_index]  # Toggle the state of the hole

# Counting the number of teeth left
num_teeth = sum(holes)

# Writing the output to stdout
sys.stdout.write(str(num_teeth))