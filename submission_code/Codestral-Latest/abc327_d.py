import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
M = int(data[1])

A = list(map(int, data[2:2+M]))
B = list(map(int, data[2+M:]))

# Create a dictionary to store the pairs (A_i, B_i)
pair_dict = {}

for i in range(M):
    pair = (A[i], B[i])
    if pair in pair_dict:
        pair_dict[pair] += 1
    else:
        pair_dict[pair] = 1

# Check if any pair (A_i, B_i) appears more than once
for count in pair_dict.values():
    if count > 1:
        print("No")
        break
else:
    print("Yes")