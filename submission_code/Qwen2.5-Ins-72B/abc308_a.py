# YOUR CODE HERE
S = list(map(int, input().split()))

# Check if the sequence is monotonically non-decreasing
if not all(S[i] <= S[i + 1] for i in range(7)):
    print("No")
    exit()

# Check if all elements are between 100 and 675 inclusive
if not all(100 <= s <= 675 for s in S):
    print("No")
    exit()

# Check if all elements are multiples of 25
if not all(s % 25 == 0 for s in S):
    print("No")
    exit()

print("Yes")