n = input().strip()

# Check if all characters are among '1', '2', '3'
if set(n).issubset({'1', '2', '3'}):
    count1 = n.count('1')
    count2 = n.count('2')
    count3 = n.count('3')
    
    # Check if counts match the required frequencies
    if count1 == 1 and count2 == 2 and count3 == 3:
        print("Yes")
    else:
        print("No")
else:
    print("No")