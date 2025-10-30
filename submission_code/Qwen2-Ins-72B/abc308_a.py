S = list(map(int, input().split()))

# Check if the sequence is monotonically non-decreasing
if not all(S[i] <= S[i+1] for i in range(len(S)-1)):
    print("No")
    exit()

# Check if all numbers are between 100 and 675, and are multiples of 25
if not all(100 <= num <= 675 and num % 25 == 0 for num in S):
    print("No")
    exit()

print("Yes")