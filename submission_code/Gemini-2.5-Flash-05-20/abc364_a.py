import sys

def solve():
    N = int(sys.stdin.readline())
    
    # Read all dish types into a list
    dishes = []
    for _ in range(N):
        dishes.append(sys.stdin.readline().strip())
        
    can_eat_all_dishes = True
    
    # Iterate through the dishes to check for consecutive sweet dishes
    # We only need to check up to N-2 because we look at S[i] and S[i+1]
    for i in range(N - 1):
        # If the current dish and the next dish are both 'sweet'
        if dishes[i] == 'sweet' and dishes[i+1] == 'sweet':
            # Takahashi gets sick after eating dishes[i+1].
            # If dishes[i+1] is NOT the very last dish (dishes[N-1]),
            # then he cannot eat all dishes because there are more dishes remaining.
            if (i + 1) < (N - 1):
                can_eat_all_dishes = False
                break # No need to check further, we already know he fails
                
    if can_eat_all_dishes:
        print("Yes")
    else:
        print("No")

# Call the solve function to run the program
solve()