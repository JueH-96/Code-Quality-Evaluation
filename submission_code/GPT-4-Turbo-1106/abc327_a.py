# Read input
N = int(input().strip())
S = input().strip()

# Check for adjacent occurrences of 'a' and 'b'
if 'ab' in S or 'ba' in S:
    print('Yes')
else:
    print('No')