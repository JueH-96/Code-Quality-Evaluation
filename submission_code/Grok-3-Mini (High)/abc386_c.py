import sys

def can_delete_to_match(A, B):
    i = 0
    j = 0
    deleted = False
    len_A = len(A)
    len_B = len(B)
    while i < len_A and j < len_B:
        if A[i] == B[j]:
            i += 1
            j += 1
        else:
            if not deleted:
                i += 1  # delete A[i]
                deleted = True
            else:
                return False
    return j == len_B

# Read input
data = sys.stdin.read().split()
K = int(data[0])
S = data[1]
T = data[2]

len_S = len(S)
len_T = len(T)
diff_len = abs(len_S - len_T)

if diff_len > K:
    print("No")
elif diff_len == 0:
    count_diff = sum(1 for a, b in zip(S, T) if a != b)
    if count_diff <= K:
        print("Yes")
    else:
        print("No")
else:  # diff_len <= K and diff_len > 0, for K=1, diff_len=1
    if len_S > len_T:
        if can_delete_to_match(S, T):
            print("Yes")
        else:
            print("No")
    else:  # len_T > len_S
        if can_delete_to_match(T, S):
            print("Yes")
        else:
            print("No")