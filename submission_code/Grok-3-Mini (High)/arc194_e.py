# YOUR CODE HERE
import sys
data = sys.stdin.read().split()
index = 0
N = int(data[index])
X = int(data[index+1])
Y = int(data[index+2])
index += 3
S = data[index]
index += 1
T = data[index]

# Count number of '1' in S and T
count_S1 = S.count('1')
count_T1 = T.count('1')

if count_S1 != count_T1:
    print("No")
else:
    if N < X + Y or X + Y > N:
        if S == T:
            print("Yes")
        else:
            print("No")
    else:
        print("Yes")