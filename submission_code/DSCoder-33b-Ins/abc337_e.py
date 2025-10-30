# YOUR CODE HERE
N = int(input())

# Initialize the list of bottles
bottles = [i for i in range(1, N+1)]

# Initialize the list of friends
friends = []

# Call friends until we have more than one bottle
while len(bottles) > 1:
    # Call a new friend
    friends.append(bottles[:len(bottles)//2])
    # Remove the bottles that were given to the friend
    bottles = bottles[len(bottles)//2:]

# Print the number of friends
print(len(friends))

# Print the bottles each friend was given
for friend in friends:
    print(len(friend), *friend)

# Get the responses from the judge
responses = input()

# Identify the spoiled bottle
spoiled_bottle = 0
for i in range(len(responses)):
    if responses[i] == '1':
        spoiled_bottle += len(friends[i])

# Print the spoiled bottle
print(spoiled_bottle)