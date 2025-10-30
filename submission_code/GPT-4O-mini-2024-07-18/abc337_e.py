# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().strip().split()
    
    N = int(data[0])  # Read the number of bottles
    M = 7  # We will use 7 friends to identify the spoiled bottle
    
    # We will distribute the bottles in a way that allows us to use binary representation
    # to identify the spoiled bottle. Each friend will represent a bit in a binary number.
    
    # Each friend will get bottles corresponding to their bit position
    friends_bottles = []
    
    for i in range(M):
        bottles = []
        for j in range(1, N + 1):
            if (j - 1) & (1 << i):  # Check if the i-th bit is set in the bottle number j
                bottles.append(j)
        friends_bottles.append(bottles)
    
    # Print the number of friends
    print(M)
    
    # Print the bottles for each friend
    for bottles in friends_bottles:
        print(len(bottles), ' '.join(map(str, bottles)))
    
    # Read the response from the judge
    response = input().strip()
    
    # Determine which bottle is spoiled based on the responses
    spoiled_bottle = 0
    for i in range(M):
        if response[i] == '1':
            spoiled_bottle |= (1 << i)  # Set the i-th bit if the friend has an upset stomach
    
    # Print the number of the spoiled bottle
    print(spoiled_bottle + 1)  # +1 because bottles are 1-indexed

if __name__ == "__main__":
    main()