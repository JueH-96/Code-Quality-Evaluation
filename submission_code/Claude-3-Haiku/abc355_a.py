# Read the input
a, b = map(int, input().split())

# Create a list of the three suspects
suspects = [1, 2, 3]

# Remove the suspects that are not the culprit based on the witnesses' memories
suspects.remove(a)
suspects.remove(b)

# Check if there is only one remaining suspect
if len(suspects) == 1:
    print(suspects[0])
else:
    print(-1)