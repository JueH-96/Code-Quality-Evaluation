from collections import Counter

# Read the input as a list of integers
original = list(map(int, input().split()))

# Check each possible value that can be added
for x in range(1, 14):
    temp = original + [x]
    cnt = Counter(temp)
    # If there are exactly two distinct numbers
    if len(cnt) == 2:
        counts = list(cnt.values())
        # Sort the counts in descending order
        counts.sort(reverse=True)
        # Check if the counts are exactly 3 and 2
        if counts == [3, 2]:
            print("Yes")
            exit()

# If no possible addition forms a Full House
print("No")