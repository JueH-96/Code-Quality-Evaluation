# Read the number of dishes
N = int(input().strip())

# Initialize a variable to track if Takahashi felt sick
felt_sick = False

# Initialize a variable to track the previous dish
previous_dish = None

# Loop through each dish
for _ in range(N):
    dish = input().strip()
    
    # Check if the current dish is sweet and the previous dish was also sweet
    if dish == "sweet" and previous_dish == "sweet":
        felt_sick = True
        break
    
    # Update the previous dish
    previous_dish = dish

# Output the result
print("No" if felt_sick else "Yes")