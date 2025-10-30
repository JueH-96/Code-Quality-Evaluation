# Read the input
N = int(input())
S = input()

# Check if there are any adjacent occurrences of 'a' and 'b'
if 'ab' in S or 'ba' in S:
    print('Yes')
else:
    print('No')