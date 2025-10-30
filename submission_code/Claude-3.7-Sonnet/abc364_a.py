# Read the number of dishes
N = int(input())

# Keep track of whether the previous dish was sweet
previous_dish_sweet = False

# Process each dish
for i in range(N):
    current_dish = input().strip()
    
    if current_dish == "sweet":
        if previous_dish_sweet:
            # Takahashi has eaten two consecutive sweet dishes
            # He feels sick and can't eat any more dishes
            if i < N - 1:
                # If there are more dishes to eat, he can't eat all
                print("No")
                exit()
            # Otherwise, he's eaten all dishes and it's fine
        previous_dish_sweet = True
    else:
        previous_dish_sweet = False

# If we reached here, he can eat all dishes
print("Yes")