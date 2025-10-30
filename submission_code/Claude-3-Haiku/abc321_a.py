n = int(input())

# Convert the number to a string and check if the digits are in strictly decreasing order
if str(n) == ''.join(sorted(str(n), reverse=True)):
    print('Yes')
else:
    print('No')