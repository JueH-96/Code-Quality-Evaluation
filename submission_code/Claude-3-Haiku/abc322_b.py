# Read the input
n, m = map(int, input().split())
s = input()
t = input()

# Check if S is a prefix of T
if s == t[:n]:
    # Check if S is also a suffix of T
    if s == t[-n:]:
        print(0)
    else:
        print(1)
# Check if S is a suffix of T
elif s == t[-n:]:
    print(2)
else:
    print(3)