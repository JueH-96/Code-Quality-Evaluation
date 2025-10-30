n = input().strip()

# Check if any digit is not 1, 2, or 3
for c in n:
    if c not in {'1', '2', '3'}:
        print("No")
        exit()

# Count the occurrences of each digit
count1 = n.count('1')
count2 = n.count('2')
count3 = n.count('3')

# Check the counts
if count1 == 1 and count2 == 2 and count3 == 3:
    print("Yes")
else:
    print("No")