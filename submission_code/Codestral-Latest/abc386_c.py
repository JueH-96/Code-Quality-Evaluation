# YOUR CODE HERE
def can_transform(S, T):
    if len(S) == len(T):
        # Check for one replacement
        diff_count = 0
        for s, t in zip(S, T):
            if s != t:
                diff_count += 1
        if diff_count == 1:
            return True
    elif len(S) + 1 == len(T):
        # Check for one insertion
        for i in range(len(S)):
            if S[:i] + S[i+1:] == T:
                return True
    elif len(S) == len(T) + 1:
        # Check for one deletion
        for i in range(len(T)):
            if S[:i] + S[i+1:] == T:
                return True
    return False

# Read input
K = int(input())
S = input().strip()
T = input().strip()

# Since K=1, we only need to check for one operation
if can_transform(S, T):
    print("Yes")
else:
    print("No")