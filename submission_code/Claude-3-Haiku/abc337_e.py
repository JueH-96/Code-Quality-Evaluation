import sys

def main():
    N = int(input())
    
    # Initialize the list of bottles to test
    bottles = list(range(1, N+1))
    
    # Determine the minimum number of friends to call
    M = 1
    while len(bottles) > 1:
        # Divide the bottles into M groups
        groups = [bottles[i::M] for i in range(M)]
        
        # Serve the bottles to the friends
        for i, group in enumerate(groups):
            print(len(group), *group)
            sys.stdout.flush()
        
        # Get the friends' responses
        response = input()
        
        # Identify the spoiled bottle
        spoiled_bottle = None
        for i, char in enumerate(response):
            if char == '1':
                spoiled_bottle = groups[i][0]
                break
        
        # Update the list of bottles to test
        bottles = [bottle for bottle in bottles if bottle != spoiled_bottle]
        
        # Increase the number of friends to call
        M += 1
    
    # Print the spoiled bottle's number
    print(bottles[0])
    sys.stdout.flush()

if __name__ == "__main__":
    main()