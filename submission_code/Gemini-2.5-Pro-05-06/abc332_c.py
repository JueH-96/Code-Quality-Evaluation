import sys

# Function to read a line and parse space-separated integers
def read_ints():
    return list(map(int, sys.stdin.readline().split()))

# Function to read a single string
def read_str():
    return sys.stdin.readline().strip()

def main():
    N, M = read_ints()
    S = read_str()
    
    # M is the number of plain T-shirts Takahashi has initially.
    # We need to find the minimum number of logo T-shirts to buy.
    
    # `current_plain_shirts` tracks the number of plain T-shirts currently clean and ready to use.
    current_plain_shirts = M
    
    # `current_logo_shirts` tracks the number of logo T-shirts (from the pool of `max_logo_shirts_needed` ones)
    # that are currently clean and ready to use.
    # It starts at 0, because `max_logo_shirts_needed` is initially 0.
    # It represents the logo shirts available from the *already committed* pool of bought shirts.
    current_logo_shirts = 0
    
    # `max_logo_shirts_needed` is the answer. It's the peak number of logo shirts needed at any point,
    # which dictates the total number that must be bought.
    max_logo_shirts_needed = 0 

    for i in range(N):
        event_type = S[i]

        if event_type == '0':
            # Wash day. All T-shirts worn so far are washed.
            # All M plain T-shirts become available again.
            current_plain_shirts = M
            # All logo T-shirts that we have determined must be bought (up to `max_logo_shirts_needed`)
            # also become available.
            current_logo_shirts = max_logo_shirts_needed
        
        elif event_type == '1': # Meal day
            # Needs a plain or logo T-shirt.
            # Prioritize using a plain T-shirt to save logo T-shirts for events.
            if current_plain_shirts > 0:
                current_plain_shirts -= 1
            # If no plain T-shirt is available, use an available logo T-shirt
            # (from the pool of `max_logo_shirts_needed` T-shirts).
            elif current_logo_shirts > 0:
                current_logo_shirts -= 1
            # If neither plain nor existing logo T-shirt is available,
            # we are forced to use a "new" logo T-shirt. This means we increment
            # our count of `max_logo_shirts_needed`.
            else:
                max_logo_shirts_needed += 1
                # This newly accounted-for T-shirt is used immediately for today's meal.
                # `current_logo_shirts` was 0. The pool of bought shirts increased by one, 
                # and that one was used. So, the number of available clean logo shirts from this pool remains 0.
        
        elif event_type == '2': # Competitive programming event day
            # Must use a logo T-shirt.
            # Try to use an available logo T-shirt (from the pool of `max_logo_shirts_needed`).
            if current_logo_shirts > 0:
                current_logo_shirts -= 1
            # If no existing logo T-shirt is available,
            # we are forced to use a "new" logo T-shirt. Increment `max_logo_shirts_needed`.
            else:
                max_logo_shirts_needed += 1
                # This newly accounted-for T-shirt is used immediately.
                # Similar to type '1', `current_logo_shirts` effectively stays 0.
                
    print(max_logo_shirts_needed)

if __name__ == '__main__':
    main()