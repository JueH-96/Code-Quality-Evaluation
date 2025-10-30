# YOUR CODE HERE
from collections import *

S = input()
cnt = Counter(Counter(S).values())

for i in range(1, max(cnt.keys()) + 1):
    if cnt[i] > 2 or (cnt[i] == 1 and i != 1):
        print("No")
        exit()
print("Yes")