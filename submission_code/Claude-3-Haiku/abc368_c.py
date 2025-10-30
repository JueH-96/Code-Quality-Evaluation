# Read the input
n = int(input())
health = [int(x) for x in input().split()]

# Initialize the time counter
t = 0

# Repeat the attack until all enemies are defeated
while any(h > 0 for h in health):
    t += 1
    # Attack the frontmost enemy
    if t % 3 == 0:
        health[0] -= 3
    else:
        health[0] -= 1
    # Move the defeated enemies to the back
    if health[0] <= 0:
        health.pop(0)
        health.append(0)

# Print the final time
print(t)