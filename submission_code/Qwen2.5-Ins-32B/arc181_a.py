# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

case_idx = 1
case_count = int(data[0])
results = []

while case_idx < len(data):
    n = int(data[case_idx])
    p = list(map(int, data[case_idx+1:case_idx+1+n]))
    case_idx += 1 + n

    # Find the first index where P[i] != i+1
    first_mismatch = next((i for i in range(n) if p[i] != i + 1), None)
    if first_mismatch is None:
        results.append(0)
        continue

    # Find the last index where P[i] != i+1
    last_mismatch = next((i for i in range(n-1, -1, -1) if p[i] != i + 1), None)

    # If there is only one mismatch, one operation is enough
    if first_mismatch == last_mismatch:
        results.append(1)
        continue

    # If the subarray between first and last mismatch is sorted,
    # then it can be fixed in one operation, otherwise two operations are needed
    subarray = p[first_mismatch:last_mismatch+1]
    if subarray == sorted(subarray):
        results.append(1)
    else:
        results.append(2)

for result in results:
    print(result)