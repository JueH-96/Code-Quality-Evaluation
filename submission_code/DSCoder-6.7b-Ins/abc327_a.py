# YOUR CODE HERE

import sys

N = int(sys.stdin.readline().strip())
S = sys.stdin.readline().strip()

if 'ab' in S or 'ba' in S:
    print('Yes')
else:
    print('No')